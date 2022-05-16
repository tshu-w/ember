from pathlib import Path

import pandas as pd

for cat in ["all", "shoes", "clothing", "accessories"]:
    print(cat)
    folder = Path(f"./data/ali/datasets/{cat}/")

for cat in ["all", "shoes", "clothing", "accessories"]:
    print(cat)
    folder = Path(f"./data/ali/datasets/{cat}/")
    train = pd.read_parquet(folder / "train.parquet")
    val = pd.read_parquet(folder / "val.parquet")
    print(len(train[train["label"] == 1]), len(train[train["label"] == 0]))
    print(len(val[val["label"] == 1]), len(val[val["label"] == 0]))
    print()
    for suffix in ["", "rl", "cfm", "om", "i", "irl", "icfm", "iom"]:
        test_filename = f"test_{suffix}.parquet" if suffix else "test.parquet"
        test = pd.read_parquet(folder / test_filename)
        print(len(test[test["label"] == 1]), len(test[test["label"] == 0]))
