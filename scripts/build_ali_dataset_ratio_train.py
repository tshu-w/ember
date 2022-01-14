#!/usr/bin/env python

import math
import os
from collections import namedtuple
from pathlib import Path
from typing import Optional

import jieba
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split
from tqdm import tqdm

seed_everything(142)

CATEGORIES = ["all", "clothing", "shoes", "accessories"]
CAT2CATE_LEVEL_NAME = {
    "all": ["女装/女士精品", "女鞋", "男装", "服饰配件/皮带/帽子/围巾", "流行男鞋", "运动服/休闲服装", "运动鞋new"],
    "clothing": ["女装/女士精品", "男装", "运动服/休闲服装"],
    "shoes": ["女鞋", "流行男鞋", "运动鞋new"],
    "accessories": ["服饰配件/皮带/帽子/围巾"],
}

Split = namedtuple("Split", ["main", "extra"])
CLUSTER_SIZE_SPLIT = Split(250, 100)
RECORD_RADIO_SPLIT = Split(0.6, 0.4)

POS_NEG = namedtuple("POS_NEG", ["pos", "neg"])
POS_NEG_SIZE = POS_NEG(1, 3)
IMBALANCE_POS_NEG_SIZE = POS_NEG(500, 99 * 500)

NUM_PAIRS = 40
NEW_RECORD_NUM_PAIRS = NEW_RECORDS_NUM_PAIRS = 8
NEW_CLUSTER_NUM_PAIRS = 20

SIMILAR_CLUSTER_NUM = 12

assert NUM_PAIRS * 0.2 == NEW_RECORD_NUM_PAIRS
# assert (
#     CLUSTER_SIZE_SPLIT.main * NUM_PAIRS * 0.2
#     == CLUSTER_SIZE_SPLIT.extra * NEW_CLUSTER_NUM_PAIRS
#     == sum(IMBALANCE_POS_NEG_SIZE)
# )

os.makedirs(os.path.expanduser("~/.cache/jieba"), exist_ok=True)
jieba.dt.tmp_dir = os.path.expanduser("~/.cache/jieba")


def jaccard_similarity(lst1: list, lst2: list) -> float:
    s1 = set(lst1)
    s2 = set(lst2)
    return len(s1 & s2) / len(s1 | s2)


def union_tokenized_title(tokenized_titles: pd.Series) -> list[str]:
    return list(set.union(*map(set, tokenized_titles)))


def get_extra_records_ids(record_ids: pd.Series) -> pd.Series:
    return Split(
        *train_test_split(record_ids, test_size=RECORD_RADIO_SPLIT.extra)
    ).extra


def build_positive_pairs(
    sub_corpus: pd.DataFrame,
    corpus: Optional[pd.DataFrame] = None,
    excluded_pairs: Optional[pd.DataFrame] = None,
    num_per_cluster: Optional[int] = None,
    total_num: Optional[int] = None,
) -> pd.DataFrame:
    pairs_lst = []

    if corpus is None:
        corpus = sub_corpus

    if num_per_cluster is None:
        assert total_num is not None
        num_clusters = len(sub_corpus["cluster_id"].unique())
        num_per_cluster = math.ceil(total_num / num_clusters)

    num_hard = num_per_cluster // 2 + num_per_cluster % 2
    num_random = num_per_cluster // 2

    for cluster_id, records in tqdm(sub_corpus.groupby("cluster_id")):
        same_cluster_records = corpus[corpus["cluster_id"] == cluster_id]
        record_cartesian = pd.merge(
            records, same_cluster_records, suffixes=("_left", "_right"), how="cross"
        )
        record_pairs = record_cartesian[
            record_cartesian["id_left"] != record_cartesian["id_right"]
        ].copy()
        if excluded_pairs is not None:
            record_pairs = pd.merge(
                record_pairs,
                excluded_pairs,
                on=["id_left", "id_right"],
                how="left",
                indicator=True,
            )
            record_pairs = record_pairs[record_pairs["_merge"] == "left_only"]
            record_pairs = record_pairs.drop(columns="_merge")

        if len(record_pairs) == 0:
            continue

        record_pairs["similarity"] = record_pairs.apply(
            lambda row: jaccard_similarity(
                row["tokenized_title_left"], row["tokenized_title_right"]
            ),
            axis=1,
        )
        record_pairs.sort_values(by="similarity", inplace=True)

        hard_pairs = record_pairs[:num_hard]
        remained_len = len(record_pairs[num_hard:])
        random_pairs = record_pairs[num_hard:].sample(n=min(num_random, remained_len))

        drop_columns = [
            "tokenized_title_left",
            "tokenized_title_right",
            "similarity",
        ]
        hard_pairs = hard_pairs.drop(columns=drop_columns)
        random_pairs = random_pairs.drop(columns=drop_columns)

        pairs_lst.append(hard_pairs)
        pairs_lst.append(random_pairs)

    pairs = pd.concat(pairs_lst, ignore_index=True)
    if total_num is not None:
        pairs = pairs.sample(n=min(total_num, len(pairs)), ignore_index=True)
    pairs["label"] = 1

    return pairs


def build_negative_pairs(
    sub_corpus: pd.DataFrame,
    corpus: Optional[pd.DataFrame] = None,
    excluded_pairs: Optional[pd.DataFrame] = None,
    num_per_cluster: Optional[int] = None,
    total_num: Optional[int] = None,
) -> pd.DataFrame:
    pairs_lst = []

    if corpus is None:
        corpus = sub_corpus

    if num_per_cluster is None:
        assert total_num is not None
        num_clusters = len(sub_corpus["cluster_id"].unique())
        num_per_cluster = math.ceil(total_num / num_clusters)

    num_hard = num_per_cluster // 2 + num_per_cluster % 2
    num_random = num_per_cluster // 2

    cluster_id_groups = corpus.groupby("cluster_id")
    cluster_title_df = (
        cluster_id_groups[["tokenized_title"]].agg(union_tokenized_title).reset_index()
    )

    for cluster_id, records in tqdm(sub_corpus.groupby("cluster_id")):
        cluster_title = union_tokenized_title(records["tokenized_title"])
        cluster_title_df["similarity"] = cluster_title_df["tokenized_title"].apply(
            lambda s: jaccard_similarity(s, cluster_title)
        )

        similar_cluster_ids = cluster_title_df[
            cluster_title_df["cluster_id"] != cluster_id
        ].sort_values(by="similarity", ascending=False)[:SIMILAR_CLUSTER_NUM][
            "cluster_id"
        ]

        similar_records = corpus[corpus["cluster_id"].isin(similar_cluster_ids)]
        record_pairs = pd.merge(
            records, similar_records, suffixes=("_left", "_right"), how="cross"
        )
        if excluded_pairs is not None:
            record_pairs = pd.merge(
                record_pairs,
                excluded_pairs,
                on=["id_left", "id_right"],
                how="left",
                indicator=True,
            )
            record_pairs = record_pairs[record_pairs["_merge"] == "left_only"]
            record_pairs = record_pairs.drop(columns="_merge")

        if len(record_pairs) == 0:
            continue

        record_pairs["similarity"] = record_pairs.apply(
            lambda row: jaccard_similarity(
                row["tokenized_title_left"], row["tokenized_title_right"]
            ),
            axis=1,
        )
        record_pairs.sort_values(by="similarity", inplace=True, ascending=False)

        hard_pairs = record_pairs[:num_hard]
        remained_len = len(record_pairs[num_hard:])
        random_pairs = record_pairs[num_hard:].sample(n=min(num_random, remained_len))

        drop_columns = [
            "tokenized_title_left",
            "tokenized_title_right",
            "similarity",
        ]
        hard_pairs = hard_pairs.drop(columns=drop_columns)
        random_pairs = random_pairs.drop(columns=drop_columns)

        pairs_lst.append(hard_pairs)
        pairs_lst.append(random_pairs)

    pairs = pd.concat(pairs_lst, ignore_index=True)
    if total_num is not None:
        pairs = pairs.sample(n=min(total_num, len(pairs)), ignore_index=True)

    pairs["label"] = 0

    return pairs


def build_record_pairs(
    sub_corpus: pd.DataFrame,
    pos_corpus: Optional[pd.DataFrame] = None,
    neg_corpus: Optional[pd.DataFrame] = None,
    excluded_pairs: Optional[pd.DataFrame] = None,
    num_per_cluster: Optional[int] = None,
    pos_neg_size: POS_NEG = POS_NEG_SIZE,
) -> pd.DataFrame:
    if num_per_cluster is not None:
        assert num_per_cluster % sum(pos_neg_size) == 0
        num_pos = num_per_cluster // sum(pos_neg_size) * pos_neg_size.pos
        num_neg = num_per_cluster // sum(pos_neg_size) * pos_neg_size.neg

        pos_pairs = build_positive_pairs(
            sub_corpus,
            pos_corpus,
            excluded_pairs,
            num_per_cluster=num_pos,
        )
        neg_pairs = build_negative_pairs(
            sub_corpus,
            neg_corpus,
            excluded_pairs,
            num_per_cluster=num_neg,
        )
    else:
        pos_pairs = build_positive_pairs(
            sub_corpus,
            pos_corpus,
            excluded_pairs,
            total_num=pos_neg_size.pos,
        )
        neg_pairs = build_negative_pairs(
            sub_corpus,
            neg_corpus,
            excluded_pairs,
            total_num=pos_neg_size.neg,
        )

    return pd.concat((pos_pairs, neg_pairs))


def build_datasets(corpus: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cluster_ids = corpus["cluster_id"].unique()
    sampled_cluster_ids = np.random.choice(
        cluster_ids, size=sum(CLUSTER_SIZE_SPLIT), replace=False
    )
    cluster_ids_split = Split(
        *train_test_split(sampled_cluster_ids, test_size=CLUSTER_SIZE_SPLIT.extra)
    )
    assert len(cluster_ids_split.main) + len(cluster_ids_split.extra) == len(
        sampled_cluster_ids
    )
    assert set(cluster_ids_split.main) & set(cluster_ids_split.extra) == set()
    cluster_split = Split(
        *(
            corpus[corpus["cluster_id"].isin(cluster_ids)]
            for cluster_ids in cluster_ids_split
        )
    )

    record_ids = cluster_split.main["id"]
    cluster_id_groups = cluster_split.main.groupby("cluster_id")
    extract_record_ids = cluster_id_groups["id"].apply(get_extra_records_ids)
    main_record_ids = record_ids[~record_ids.isin(extract_record_ids)]
    assert len(main_record_ids) + len(extract_record_ids) == len(record_ids)
    assert set(main_record_ids) & set(extract_record_ids) == set()
    record_ids_split = Split(main_record_ids, extract_record_ids)
    record_split = Split(
        *(corpus[corpus["id"].isin(record_ids)] for record_ids in record_ids_split)
    )

    datasets = {}

    for k in [9, 19, 39, 79, 99]:
        assert 2000 % (k + 1) == 0
        imbalance_pos_neg_size = POS_NEG(10000 // (k + 1), 10000 * k // (k + 1))

        dataset = build_record_pairs(
            record_split.main, pos_neg_size=imbalance_pos_neg_size
        )

        # 7:1:2
        train_val, test = train_test_split(dataset, test_size=0.2)
        train, val = train_test_split(train_val, test_size=1 / 8)

        imbalance_pos_neg_size = POS_NEG(2000 // (k + 1), 2000 * k // (k + 1))
        new_record_test = build_record_pairs(
            record_split.main,
            pos_corpus=record_split.extra,
            neg_corpus=record_split.extra,
            pos_neg_size=imbalance_pos_neg_size,
        )
        new_records_test = build_record_pairs(
            record_split.extra,
            pos_neg_size=imbalance_pos_neg_size,
        )
        new_cluster_test = build_record_pairs(
            cluster_split.extra,
            pos_neg_size=imbalance_pos_neg_size,
        )

        datasets |= {
            f"train_{k+1}": train,
            f"val_{k+1}": val,
            f"test_{k+1}": test,
            f"test_nr_{k+1}": new_record_test,
            f"test_nrs_{k+1}": new_records_test,
            f"test_nc_{k+1}": new_cluster_test,
        }

    return datasets


def main():
    corpus = pd.read_parquet("./data/ali/corpus.parquet")
    corpus["tokenized_title"] = corpus["title"].apply(
        lambda s: list(filter(lambda x: x.strip(), jieba.cut(s))),
    )

    for cat in CATEGORIES:
        print(cat)
        cat_corpus = corpus[corpus["cate_level_name"].isin(CAT2CATE_LEVEL_NAME[cat])]

        # filter clusters less than 10 in the cat_corpus
        lt10_bool = cat_corpus["cluster_id"].value_counts() < 10
        clusters_lt10 = lt10_bool[lt10_bool].index
        cat_corpus = cat_corpus[~cat_corpus["cluster_id"].isin(clusters_lt10)]

        datasets = build_datasets(cat_corpus)
        datasets_path = Path(f"./data/ali/datasets_ratio/{cat}/")
        datasets_path.mkdir(parents=True, exist_ok=True)
        for k, df in datasets.items():
            df.to_parquet(datasets_path / f"{k}.parquet", index=False)


if __name__ == "__main__":
    main()
