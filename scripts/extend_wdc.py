#!/usr/bin/env python

import random
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.similarities import Similarity, SparseMatrixSimilarity
from pytorch_lightning import seed_everything
from tqdm import tqdm

DATA_DIR = Path("./data/wdc/norm")
CATEGORY_MAP = {
    "cameras": "Camera_and_Photo",
    "computers": "Computers_and_Accessories",
    "shoes": "Shoes",
    "watches": "Jewelry",
}
seed_everything(142)


# from https://github.com/ir-ischool-uos/mwpd/blob/9e4c63e0ac08b9491cdd91feb6ed14e19bc266cb/prodmatch/processing/sample-training-set.ipynb
def build_positive_pairs(corpus, clusters, attribute, num_pos):
    """
    Builds positive pairs for all offers in each cluster in 'clusters'
    which can be found in 'corpus' using 'attribute' for calculating
    BOW cosine similarity to select hard pairs.
    Selects an equal amount of hard and random pairs depending on 'num_pos'
    per offer. If it is not possible to build 'num_pos' pairs, the heuristic
    will build as many pairs as possible for that cluster.

    Parameters:
    corpus (pandas.DataFrame): Corpus containing product offers.
    clusters (List): List of cluster_ids for which Positive pairs should be built.
    attribute (str): Attribute of 'corpus' to use for similarity calculations.
    num_pos (int): Number of positive examples to build per offer.

    Returns:
    List(Tuple(int, List(List,List))): a list of tuples, each tuple containing
    the offer id and a list of two lists containing the offer ids of the hard
    and random pairs.
    """
    pos_pairs = []
    for current_cluster in tqdm(clusters):
        cluster_data = corpus[corpus["cluster_id"] == current_cluster]

        # build gensim dictionary, corpus and search index for selected cluster
        dct = Dictionary(cluster_data[attribute], prune_at=5000000)
        dct.filter_extremes(no_below=2, no_above=1.0, keep_n=None)
        gensim_corpus = [dct.doc2bow(text) for text in cluster_data[attribute]]
        index = SparseMatrixSimilarity(
            gensim_corpus, num_features=len(dct), num_best=80
        )

        # query up to 80 most similar offers, only offers with similarity > 0 will be returned
        query = index[gensim_corpus]

        for i, offer_sim_dup in enumerate(query):

            current_num_pos = num_pos
            current_id = cluster_data.iloc[i]["id"]

            offer_sim = []

            # remove self
            for x in offer_sim_dup:
                if x[0] != i:
                    offer_sim.append(x)

            # check if any pairs > 0 similarity remain
            if len(offer_sim) == 0:
                pos_pairs.append((current_id, [[], []]))
                continue

            # adapt number of selectable pairs if too few available
            offer_len = len(offer_sim)
            if offer_len < current_num_pos:
                current_num_pos = offer_len

            if current_num_pos == 1:
                hard_pos = 1
                random_pos = 0
            elif current_num_pos % 2 == 1:
                hard_pos = int(current_num_pos / 2) + 1
                random_pos = int(current_num_pos / 2)
            else:
                hard_pos = int(current_num_pos / 2)
                random_pos = int(current_num_pos / 2)

            # get hard offers from bottom of list
            hard_offers = offer_sim[-hard_pos:]

            if random_pos == 0:
                pos_pairs.append(
                    (
                        current_id,
                        [[cluster_data.iloc[x[0]]["id"] for x in hard_offers], []],
                    )
                )
                continue

            # remaining offers
            rest = offer_sim[:-hard_pos]

            # randomly select from remaining
            random_select = random.sample(range(len(rest)), random_pos)
            random_offers = [rest[idx] for idx in random_select]

            hard_ids = [cluster_data.iloc[x[0]]["id"] for x in hard_offers]
            random_ids = [cluster_data.iloc[x[0]]["id"] for x in random_offers]

            pos_pairs.append((current_id, [hard_ids, random_ids]))
    return pos_pairs


# def build_neg_pairs_for_cat(corpus, category, offers, attribute, num_neg):
def build_neg_pairs(corpus, offers, attribute, num_neg):
    """
    Builds negative pairs for all offers in 'offers' which are of category
    'category' which can be found in 'corpus' using 'attribute' for calculating
    BOW cosine similarity to select hard pairs.
    Selects an equal amount of hard and random pairs depending on 'num_neg'
    per offer. Each hard negative will originate from a different cluster
    to avoid building hard negatives with only a small amount of different
    products. If offers in 'offers' originate from multiple categories,
    this function should be called multiple times while iterating over
    the different categories.

    Parameters:
    corpus (pandas.DataFrame): Corpus containing product offers
    category (str): Category for which to build negatives
    offers (List): List of offer_ids for which to build negatives
    attribute (str): Attribute of 'corpus' to use for similarity calculations
    num_neg (int): Number of negative examples to build per offer

    Returns:
    List(Tuple(int, List(List,List))): a list of tuples, each tuple containing
    the offer id and a list of two lists containing the offer ids of the hard
    and random pairs.
    """
    # select data from relevant category
    # cat_data = corpus[corpus["category"] == category].copy()
    cat_data = corpus.copy()
    cat_data = cat_data.reset_index(drop=True)
    cat_data["subindex"] = list(cat_data.index)

    # build gensim dictionary, corpus and search index for selected cluster
    dct = Dictionary(cat_data[attribute], prune_at=5000000)
    dct.filter_extremes(no_below=2, no_above=0.8, keep_n=None)

    gensim_corpus = [dct.doc2bow(text) for text in cat_data[attribute]]

    index = Similarity(None, gensim_corpus, num_features=len(dct), num_best=200)

    # corpus to select negatives against
    corpus_neg_all = cat_data

    # corpus containing only offers for which negatives should be built
    corpus_neg = corpus_neg_all[corpus_neg_all["id"].isin(offers)]

    neg_pairs_cat = []

    # query for 200 most similar offers across whole category
    query_corpus = [gensim_corpus[i] for i in list(corpus_neg["subindex"])]
    start = time.time()
    query = index[query_corpus]
    end = time.time()
    print(f"Query took {end-start} seconds")

    for i, offer_sim in enumerate(tqdm(query)):

        current_index = corpus_neg.iloc[i]["subindex"]
        current_id = corpus_neg.iloc[i]["id"]
        current_cluster_id = corpus_neg.iloc[i]["cluster_id"]
        current_num_neg = num_neg

        # remove any offers with similarity 1.0
        sim_indices = []
        for x in offer_sim:
            if x[1] >= 1.0:
                continue
            else:
                sim_indices.append(x[0])

        possible_pairs = corpus_neg_all.loc[sim_indices]

        # filter by cluster_id, i.e. only 1 offer per cluster remains to allow for product diversity
        idx = sorted(np.unique(possible_pairs["cluster_id"], return_index=True)[1])

        possible_pairs = possible_pairs.iloc[idx]

        # remove any offer from same cluster
        possible_pairs = possible_pairs[
            possible_pairs["cluster_id"] != current_cluster_id
        ]

        possible_pairs_len = len(possible_pairs)

        # check if any pairs > 0 similarity remain
        if possible_pairs_len == 0:
            neg_pairs_cat.append((current_id, [[], []]))
            continue

        # adapt number of selectable pairs if too few available
        if possible_pairs_len < current_num_neg:
            current_num_neg = possible_pairs_len

        if current_num_neg == 1:
            hard_neg = 1
            random_neg = 0
        elif current_num_neg % 2 == 1:
            hard_neg = int(current_num_neg / 2) + 1
            random_neg = int(current_num_neg / 2)
        else:
            hard_neg = int(current_num_neg / 2)
            random_neg = int(current_num_neg / 2)

        # select hard pairs from top of list
        candidates = possible_pairs.iloc[:hard_neg]

        hard_pairs = candidates["id"].tolist()

        if random_neg == 0:
            neg_pairs_cat.append((current_id, [hard_pairs, []]))
            continue
        else:
            remove = list(candidates.index)
            remove.append(current_index)

            # randomly select from all offers among same category
            random_select = random.sample(range(len(corpus_neg_all)), random_neg)
            random_pairs = corpus_neg_all.iloc[random_select]
            while any(random_pairs["id"].isin(remove)) or any(
                random_pairs["cluster_id"] == current_cluster_id
            ):
                random_select = random.sample(range(len(corpus_neg_all)), random_neg)
                random_pairs = corpus_neg_all.iloc[random_select]
            random_pairs = random_pairs["id"].tolist()

            combined_pairs = [hard_pairs, random_pairs]
        neg_pairs_cat.append((current_id, combined_pairs))

    return neg_pairs_cat


def get_order_pairs(
    df1: pd.DataFrame, df2: pd.DataFrame, index: str = "id"
) -> pd.DataFrame:
    df_cartesian = pd.merge(df1, df2, suffixes=("_left", "_right"), how="across")

    return df_cartesian[df_cartesian[index + "_left"] != df_cartesian[index + "_right"]]


def gen_order_pairs(
    corpus: pd.DataFrame, pairs: list[tuple], label: int
) -> Iterator[pd.DataFrame]:
    """Yield order pairs."""
    for pair in pairs:
        order_id = pair[0]
        hard_ids = pair[1][0]
        random_ids = pair[1][1]

        left_orders = corpus[corpus["id"] == order_id].drop(columns=["title_tokenized"])
        right_orders = corpus[corpus["id"].isin(hard_ids + random_ids)].drop(
            columns=["title_tokenized"]
        )
        order_pairs = get_order_pairs(left_orders, right_orders)

        # order_pairs = order_pairs.rename(columns=rename_column)
        order_pairs["label"] = label
        order_pairs["pair_id"] = (
            order_pairs["id_left"].map(str) + "#" + order_pairs["id_right"].map(str)
        )

        yield order_pairs


def get_pos_and_neg_df(
    corpus: pd.DataFrame,
    cluster_num: Optional[int] = None,
    num_pos: int = 1,
    num_neg: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # tokenize title for use in similarity computations
    tokenized_title = corpus["title"].str.split()
    corpus["title_tokenized"] = tokenized_title

    # select clusters with size > 1
    gt1_bool = corpus["cluster_id"].value_counts() > 1
    clusters_gt1 = list(gt1_bool[gt1_bool == True].index)
    random_clusters = random.sample(clusters_gt1, cluster_num)

    pos_df = pd.DataFrame()
    pos_pairs = build_positive_pairs(
        corpus, random_clusters, attribute="title_tokenized", num_pos=num_pos
    )
    for order_pairs in tqdm(gen_order_pairs(corpus, pos_pairs, label=1)):
        pos_df = pos_df.append(order_pairs, ignore_index=True)

    neg_df = pd.DataFrame()
    offers_for_negatives = [x[0] for x in pos_pairs]
    neg_pairs = build_neg_pairs(
        corpus, offers_for_negatives, attribute="title_tokenized", num_neg=num_neg
    )
    for order_pairs in tqdm(gen_order_pairs(corpus, neg_pairs, label=0)):
        neg_df = neg_df.append(order_pairs, ignore_index=True)

    return pos_df, neg_df


def build_dataset(
    corpus: pd.DataFrame,
    cluster_num: int = 150,
    category: Optional[str] = "Computers_and_Accessories",
) -> pd.DataFrame:
    corpus = corpus[corpus["category"] == category] if category else corpus
    df = pd.concat(
        get_pos_and_neg_df(corpus.copy(), cluster_num=cluster_num), ignore_index=True
    )

    return df


def build_testset(
    corpus: pd.DataFrame,
    category: Optional[str] = "Computers_and_Accessories",
    cluster_num: int = 150,
    total_pos: Optional[int] = None,
    total_neg: Optional[int] = None,
) -> pd.DataFrame:
    corpus = corpus[corpus["category"] == category] if category else corpus
    pos_df, neg_df = get_pos_and_neg_df(
        corpus.copy(), cluster_num=cluster_num, num_pos=2, num_neg=6
    )
    pos_df = pos_df[:total_pos] if total_pos else pos_df
    neg_df = neg_df[:total_neg] if total_neg else pos_df

    return pd.concat([pos_df, neg_df], ignore_index=True)


def main():
    test_sets_dir = DATA_DIR / "test-sets"
    test_sets_dir.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_json(DATA_DIR / "offers_corpus_english_v2.json.gz", lines=True)
    corpus = corpus[~corpus["title"].isnull()].reset_index(drop=True)

    dfs = []
    for cat in ["cameras", "computers", "shoes", "watches"]:
        train_path = DATA_DIR / f"training-sets/{cat}_train/{cat}_train_xlarge.json.gz"
        train_df = pd.read_json(train_path, lines=True)

        cluster_ids = pd.concat(
            [train_df["cluster_id_left"], train_df["cluster_id_right"]],
            ignore_index=True,
        ).unique()

        filtered_corpus = corpus[~corpus["cluster_id"].isin(cluster_ids)]

        df = build_testset(
            filtered_corpus, category=CATEGORY_MAP[cat], total_pos=300, total_neg=800
        )
        dfs.append(df)
        df.to_json(
            test_sets_dir / f"{cat}_test.json.gz",
            orient="records",
            lines=True,
            force_ascii=False,
        )

    df = pd.concat(dfs, ignore_index=True)
    df.to_json(
        test_sets_dir / f"all_test.json.gz",
        orient="records",
        lines=True,
        force_ascii=False,
    )


if __name__ == "__main__":
    main()