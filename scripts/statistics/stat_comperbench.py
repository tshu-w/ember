from pathlib import Path

import numpy as np
import pandas as pd
from rich import print

for folder in Path("./data/comperbench/").iterdir():
    print(folder.stem)
    train = pd.read_csv(folder / "gs_train.csv")
    val = pd.read_csv(folder / "gs_val.csv")
    test = pd.read_csv(folder / "gs_test.csv")

    union_df = pd.concat((train, val, test))
    print(", ".join(map(str, union_df["matching"].value_counts().to_dict().values())))

    train_ids = np.concatenate(
        [
            train["source_id"].unique(),
            train["target_id"].unique(),
            val["source_id"].unique(),
            val["target_id"].unique(),
        ]
    )
    test["cnt"] = test["source_id"].isin(train_ids).astype(int) + test[
        "target_id"
    ].isin(train_ids).astype(int)
    cnt = test["cnt"].value_counts().to_dict()
    print(cnt)
    print((sum(k * v for k, v in cnt.items())) / (sum(cnt.values()) * 2))
