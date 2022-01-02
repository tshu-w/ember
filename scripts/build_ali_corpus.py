#!/usr/bin/env python

from pathlib import Path

import pandas as pd


def main():
    column_names = [
        "id",
        "title",
        "pict_url",
        "cate_name",
        "cate_level_name",
        "pv_pairs",
        "cluster_id",
    ]
    df = pd.read_csv(
        "../data/ali/same_product_train_sample_1wpid_USTC.txt",
        header=None,
        sep="@;@",
        names=column_names,
        engine="python",
    )
    ids = {int(i.stem) for i in Path("../data/ali/images/").glob("*.jpg")}

    df = df[df["id"].isin(ids)].reset_index(drop=True)
    df["pv_pairs"] = df["pv_pairs"].fillna("")
    # df["pv_pairs"] = df["pv_pairs"].map(
    #     lambda pv_pairs: dict(pv.split("#:#") for pv in pv_pairs.split("#;#"))
    #     if pv_pairs
    #     else {}
    # )

    df.to_parquet("../data/ali/corpus.parquet")


if __name__ == "__main__":
    main()
