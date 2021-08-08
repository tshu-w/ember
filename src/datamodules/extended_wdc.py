#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from pathlib import Path

import pandas as pd
from pytorch_lightning import seed_everything

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / Path("../../../data/wdc/")


def swap_id(pair_id):
    splited = pair_id.split("#")

    return f"{splited[1]}#{splited[0]}"


def main():
    ids = set(map(lambda x: int(x.stem[:-2]), (DATA_DIR / "images").glob("*")))

    for dir in ["nonnorm", "norm"]:
        test_sets = DATA_DIR / dir / "test-sets"
        test_sets.mkdir(parents=True, exist_ok=True)

        for cate in ["all", "cameras", "computers", "shoes", "watches"]:
            training_sets = DATA_DIR / dir / "training-sets" / f"{cate}_train"
            gold_standards = DATA_DIR / dir / "gold-standards" / f"{cate}_gs.json.gz"

            xlarge_path = list(training_sets.rglob("*_xlarge.json.gz"))[0]
            xlarge_df = pd.read_json(xlarge_path, lines=True)
            for f in itertools.chain(
                training_sets.rglob("*.json.gz"), [gold_standards]
            ):
                if "xlarge" not in str(f):
                    df = pd.read_json(f, lines=True)
                    tmp = df[~df["pair_id"].isin(xlarge_df["pair_id"])]
                    tmp = tmp[~tmp["pair_id"].isin(xlarge_df["pair_id"].apply(swap_id))]

                    xlarge_df = xlarge_df[~xlarge_df["pair_id"].isin(df["pair_id"])]
                    xlarge_df = xlarge_df[
                        ~xlarge_df["pair_id"].isin(df["pair_id"].apply(swap_id))
                    ]

            test_set = pd.read_json(gold_standards, lines=True)
            test_set = test_set.append(
                xlarge_df.sample(n=len(test_set)), ignore_index=True
            )
            test_set.to_json(
                test_sets / f"{cate}_test.json.gz",
                orient="records",
                lines=True,
                force_ascii=False,
            )


if __name__ == "__main__":
    seed_everything(123)
    main()
