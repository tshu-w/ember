import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import deepmatcher as dm
import jieba
import pandas as pd
import pyprind
import torch
from deepmatcher.data import MatchingDataset, MatchingIterator
from deepmatcher.data.process import _make_fields
from deepmatcher.runner import Runner, Statistics
from deepmatcher.utils import tally_parameters
from torchtext.utils import unicode_csv_reader
from tqdm import tqdm

os.makedirs(os.path.expanduser("~/.cache/jieba"), exist_ok=True)
jieba.dt.tmp_dir = os.path.expanduser("~/.cache/jieba")

# Return f1, precision, recall when evaluation
@staticmethod
def _run(
    run_type,
    model,
    dataset,
    criterion=None,
    optimizer=None,
    train=False,
    device=None,
    batch_size=32,
    batch_callback=None,
    epoch_callback=None,
    progress_style="bar",
    log_freq=5,
    sort_in_buckets=None,
    return_predictions=False,
    return_all_metrics=False,
    **kwargs,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "gpu":
        device = "cuda"

    sort_in_buckets = train
    run_iter = MatchingIterator(
        dataset,
        model.meta,
        train,
        batch_size=batch_size,
        device=device,
        sort_in_buckets=sort_in_buckets,
    )

    model = model.to(device)
    if criterion:
        criterion = criterion.to(device)

    if train:
        model.train()
    else:
        model.eval()

    epoch = model.epoch
    datatime = 0
    runtime = 0
    cum_stats = Statistics()
    stats = Statistics()
    predictions = []
    id_attr = model.meta.id_field
    label_attr = model.meta.label_field

    if train and epoch == 0:
        print("* Number of trainable parameters:", tally_parameters(model))

    epoch_str = f"Epoch {epoch + 1:d}"
    print("===> ", run_type, epoch_str)
    batch_end = time.time()

    # The tqdm-bar for Jupyter notebook is under development.
    if progress_style == "tqdm-bar":
        pbar = tqdm(
            total=len(run_iter) // log_freq,
            bar_format="{l_bar}{bar}{postfix}",
            file=sys.stdout,
        )

    # Use the pyprind bar as the default progress bar.
    if progress_style == "bar":
        pbar = pyprind.ProgBar(len(run_iter) // log_freq, bar_char="█", width=30)

    for batch_idx, batch in enumerate(run_iter):
        batch_start = time.time()
        datatime += batch_start - batch_end

        output = model(batch)

        # from torchviz import make_dot, make_dot_from_trace
        # dot = make_dot(output.mean(), params=dict(model.named_parameters()))
        # pdb.set_trace()

        loss = float("NaN")
        if criterion:
            loss = criterion(output, getattr(batch, label_attr))

        if hasattr(batch, label_attr):
            scores = Runner._compute_scores(output, getattr(batch, label_attr))
        else:
            scores = [0] * 4

        cum_stats.update(float(loss), *scores)
        stats.update(float(loss), *scores)

        if return_predictions:
            for idx, id in enumerate(getattr(batch, id_attr)):
                predictions.append((id, float(output[idx, 1].exp())))

        if (batch_idx + 1) % log_freq == 0:
            if progress_style == "log":
                Runner._print_stats(
                    run_type, epoch + 1, batch_idx + 1, len(run_iter), stats, cum_stats
                )
            elif progress_style == "tqdm-bar":
                pbar.update()
                Runner._set_pbar_status(pbar, stats, cum_stats)
            elif progress_style == "bar":
                pbar.update()
            stats = Statistics()

        if train:
            model.zero_grad()
            loss.backward()

            if not optimizer.params:
                optimizer.set_parameters(model.named_parameters())
            optimizer.step()

        batch_end = time.time()
        runtime += batch_end - batch_start

    if progress_style == "tqdm-bar":
        pbar.close()
    elif progress_style == "bar":
        sys.stderr.flush()

    Runner._print_final_stats(epoch + 1, runtime, datatime, cum_stats)

    if return_predictions:
        return predictions
    elif return_all_metrics:
        return {
            "f1": cum_stats.f1(),
            "prc": cum_stats.precision(),
            "rec": cum_stats.recall(),
        }
    else:
        return cum_stats.f1()


Runner._run = _run


def process_labeled(path, trained_model, ignore_columns=None):
    """Creates a dataset object for an labeled dataset.

    Args:
        path (string):
            The full path to the unlabeled data file (not just the directory).
        trained_model (:class:`~deepmatcher.MatchingModel`):
            The trained model. The model is aware of the configuration of the training
            data on which it was trained, and so this method reuses the same
            configuration for the unlabeled data.
        ignore_columns (list):
            A list of columns to ignore in the unlabeled CSV file.
    """
    with open(path, encoding="utf8") as f:
        header = next(unicode_csv_reader(f))

    train_info = trained_model.meta
    if ignore_columns is None:
        ignore_columns = train_info.ignore_columns
    column_naming = dict(train_info.column_naming)

    fields = _make_fields(
        header,
        column_naming["id"],
        column_naming["label"],
        ignore_columns,
        train_info.lowercase,
        train_info.tokenize,
        train_info.include_lengths,
    )

    dataset_args = {"fields": fields, "column_naming": column_naming}
    dataset = MatchingDataset(path=path, **dataset_args)

    # Make sure we have the same attributes.
    assert set(dataset.all_text_fields) == set(train_info.all_text_fields)

    reverse_fields_dict = {pair[1]: pair[0] for pair in fields}
    for field, name in reverse_fields_dict.items():
        if field is not None and field.use_vocab:
            # Copy over vocab from original train data.
            field.vocab = copy.deepcopy(train_info.vocabs[name])
            # Then extend the vocab.
            field.extend_vocab(
                dataset,
                vectors=train_info.embeddings,
                cache=train_info.embeddings_cache,
            )

    dataset.vocabs = {
        name: dataset.fields[name].vocab for name in train_info.all_text_fields
    }

    return dataset


def argument_parser():
    parser = argparse.ArgumentParser(description="run deepmatcher")
    parser.add_argument("--dir", help="datasets directory")
    parser.add_argument("--gpu", type=int, help="cuda device number")
    parser.add_argument("--output", help="result directory")

    parser.add_argument("--no-fit", action="store_true", help="whether not fitting")
    parser.add_argument("--test-file", default="test", help="the stem of test file")
    parser.add_argument("--model-dir", help="directory of best_model.pth")

    return parser


def rename_column(s: str):
    return "_".join(s.rsplit("_", 1)[::-1])


def preprocess(df: pd.DataFrame):
    df.rename(columns=rename_column, inplace=True)

    df["left_pv_pairs"] = df["left_pv_pairs"].str.replace("#:#|#;#", " ", regex=True)
    df["right_pv_pairs"] = df["right_pv_pairs"].str.replace("#:#|#;#", " ", regex=True)

    for prefix in ["left", "right"]:
        for column in ["title", "pv_pairs"]:
            df[f"{prefix}_{column}"] = df[f"{prefix}_{column}"].apply(
                lambda s: " ".join(filter(lambda x: x.strip(), jieba.cut(s)))
            )

    df.drop(columns=["left_pict_url", "right_pict_url"], inplace=True)

    return df


def run(args):
    dir = Path(args.dir)
    o_dir = Path(args.output)
    o_dir.mkdir(parents=True, exist_ok=True)
    for stem in ["train", "val", args.test_file]:
        if not (dir / f"{stem}.csv").exists():
            df = pd.read_parquet(dir / f"{stem}.parquet")
            df = preprocess(df)
            df.to_csv(dir / f"{stem}.csv", index_label="id")

    device = f"cuda:{args.gpu}"
    model = dm.MatchingModel()

    if args.no_fit:
        model.load_state(Path(args.model_dir) / "best_model.pth")
        test = process_labeled(
            path=str(dir / f"{args.test_file}.csv"),
            trained_model=model,
            ignore_columns=(
                "left_id",
                "right_id",
                "left_cluster_id",
                "right_cluster_id",
            ),
        )
        model = copy.deepcopy(model)
        model._reset_embeddings(test.vocabs)
    else:
        train, val, test = dm.data.process(
            path=args.dir,
            train="train.csv",
            validation="val.csv",
            test=f"{args.test_file}.csv",
            ignore_columns=(
                "left_id",
                "right_id",
                "left_cluster_id",
                "right_cluster_id",
            ),
            cache=None,
            embeddings="fasttext.zh.bin",
        )
        model.run_train(
            train,
            val,
            best_save_path=o_dir / "best_model.pth",
            device=device,
            progress_style="log",
        )

    results = Runner._run(
        "EVAL",
        model,
        test,
        device=device,
        progress_style="log",
        return_all_metrics=True,
    )
    for k, v in results.items():
        results[k] = v.item() / 100
    results_str = json.dumps(results, ensure_ascii=False, indent=2)

    metrics_file = o_dir / "metrics.json"
    with metrics_file.open("w") as f:
        f.write(results_str)


if __name__ == "__main__":
    args = argument_parser().parse_args()
    run(args)
