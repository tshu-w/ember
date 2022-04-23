#!/usr/bin/env python

from pathlib import Path

import pandas as pd

for cat in ["all", "shoes", "clothing", "accessories"]:
    print(cat)
    for suffix in ["", "rl", "cfm", "om", "i", "irl", "icfm", "iom"]:
        folder = Path(f"./data/ali/datasets/{cat}/")
        test_filename = f"test_{suffix}.parquet" if suffix else "test.parquet"
        test = pd.read_parquet(folder / test_filename)
        print(len(test[test["label"] == 1]), len(test[test["label"] == 0]))
