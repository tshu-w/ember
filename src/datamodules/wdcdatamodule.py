import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.features.features import Features
from datasets.load import load_dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)


class WDCDataModule(LightningDataModule):
    def __init__(
        self,
        cat: Literal["all", "cameras", "computers", "shoes", "watches"] = "all",
        train_size: Literal["small", "medium", "large", "xlarge"] = "medium",
        columns: list[str] = ["title"],
        extra_test: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cat = cat
        self.train_size = train_size
        self.columns = columns
        self.extra_test = extra_test

        self.data_path = Path("./data/wdc/")
        self.train_path = (
            self.data_path
            / f"norm/training-sets/{self.cat}_train/{self.cat}_train_{self.train_size}.parquet"
        )
        self.valid_path = (
            self.data_path
            / f"norm/validation-sets/{self.cat}_valid/{self.cat}_valid_{self.train_size}.parquet"
        )
        self.test_path = self.data_path / f"norm/gold-standards/{self.cat}_gs.parquet"

        # extra test
        if self.extra_test:
            self.test_path = self.data_path / f"norm/test-sets/{self.cat}_test.parquet"

        # image
        self.id2imgs = defaultdict(list)
        for f in (self.data_path / "images").glob("*_*"):
            self.id2imgs[int(f.stem.split("_")[0])].append(f)

        for k, v in self.id2imgs.items():
            self.id2imgs[k] = sorted(v)

    def prepare_data(self) -> None:
        # NOTE: Use pandas to load to avoid ArrowNotImplementedError:
        # Unsupported cast from struct to struct using function cast_struct
        remove_columns = [
            "specTableContent_left",
            "keyValuePairs_left",
            "cluster_id_left",
            "identifiers_left",
            "specTableContent_right",
            "keyValuePairs_right",
            "cluster_id_right",
            "identifiers_right",
        ]

        if not (self.train_path.exists() and self.valid_path.exists()):
            train_path = self.train_path.with_suffix(".json.gz")
            valid_path = self.valid_path.with_suffix(".csv")

            train_valid_df = pd.read_json(train_path, lines=True)
            valid_pair_id = pd.read_csv(valid_path)["pair_id"]

            train_df = train_valid_df[~train_valid_df["pair_id"].isin(valid_pair_id)]
            train_df.reset_index(drop=True, inplace=True)
            train_dataset = Dataset.from_pandas(train_df).remove_columns(remove_columns)
            train_dataset.to_parquet(self.train_path)

            valid_df = train_valid_df[train_valid_df["pair_id"].isin(valid_pair_id)]
            valid_df.reset_index(drop=True, inplace=True)
            valid_dataset = Dataset.from_pandas(valid_df).remove_columns(remove_columns)
            valid_dataset.to_parquet(self.valid_path)

        if not self.test_path.exists():
            test_path = self.test_path.with_suffix(".json.gz")
            test_dataset = Dataset.from_pandas(
                pd.read_json(test_path, lines=True)
            ).remove_columns(remove_columns)
            test_dataset.to_parquet(self.test_path)

        self.setup()  # avoid cache conflict in multi processes

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "datasets"):
            convert_to_features = self.trainer.model.convert_to_features
            features = getattr(self.trainer.model, "features", None)
            preprocess_fn = partial(
                self._preprocess, columns=self.columns, id2imgs=self.id2imgs
            )
            preprocess = lambda x: convert_to_features(preprocess_fn(x))

            datasets = load_dataset(
                "parquet",
                data_files={
                    "train": str(self.train_path),
                    "valid": str(self.valid_path),
                    "test": str(self.test_path),
                },
            )
            self.datasets = datasets.map(
                preprocess,
                batched=True,
                remove_columns=datasets["train"].column_names,
                features=Features(features) if features else None,
            )
            self.datasets.set_format(type="torch")

        self.collate_fn = getattr(self.trainer.model, "collate_fn", None)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["valid"],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    @staticmethod
    def _preprocess(batch, columns: list[str], id2imgs: dict):
        text_left = []
        for attrs in zip(*(batch[f"{c}_left"] for c in columns)):
            text_left.append(" ".join(map(lambda x: str(x or ""), attrs)))

        text_right = []
        for attrs in zip(*(batch[f"{c}_right"] for c in columns)):
            text_right.append(" ".join(map(lambda x: str(x or ""), attrs)))

        image_left = [id2imgs[i][0] if id2imgs[i] else None for i in batch["id_left"]]
        image_right = [id2imgs[i][0] if id2imgs[i] else None for i in batch["id_right"]]

        return {
            "text_left": text_left,
            "text_right": text_right,
            "image_left": image_left,
            "image_right": image_right,
            "labels": batch["label"],
        }
