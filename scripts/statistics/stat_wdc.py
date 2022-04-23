from pathlib import Path

import numpy as np
import pandas as pd
from rich import print

folder = Path("./data/wdc/norm/")

for cat in ["cameras", "computers", "shoes", "watches"]:
    print(cat)
    train_val = pd.read_json(
        folder / "training-sets" / f"{cat}_train" / f"{cat}_train_xlarge.json.gz",
        lines=True,
    )
    test = pd.read_json(folder / "gold-standards" / f"{cat}_gs.json.gz", lines=True)

    ids = np.concatenate(
        [train_val["id_left"].unique(), train_val["id_right"].unique()]
    )
    cluster_ids = np.concatenate(
        [train_val["cluster_id_left"].unique(), train_val["cluster_id_right"].unique()]
    )
    test["id_cnt"] = test["id_left"].isin(ids).astype(int) + test["id_right"].isin(
        ids
    ).astype(int)

    test["cluster_id_cnt"] = test["cluster_id_left"].isin(cluster_ids).astype(
        int
    ) + test["cluster_id_right"].isin(cluster_ids).astype(int)

    # union_df = pd.concat((train, val, test))
    # print(", ".join(map(str, union_df["matching"].value_counts().to_dict().values())))

    # train_ids = np.concatenate(
    #     [
    #         train["source_id"].unique(),
    #         train["target_id"].unique(),
    #         val["source_id"].unique(),
    #         val["target_id"].unique(),
    #     ]
    # )
    # test["cnt"] = test["source_id"].isin(train_ids).astype(int) + test[
    #     "target_id"
    # ].isin(train_ids).astype(int)
    # # print(sum(cnt.values()))
    # n_clusters = seen_clusters = 0
    # for idx, row in test.iterrows():
    #     label = int(row["matching"])
    #     if label == 1:
    #         n_clusters += 1
    #         if row["source_id"] in train_ids or row["target_id"] in train_ids:
    #             seen_clusters += 1
    #     else:
    #         n_clusters += 2
    #         if row["source_id"] in train_ids:
    #             seen_clusters += 1
    #         if row["target_id"] in train_ids:
    #             seen_clusters += 1

    cnt = test["id_cnt"].value_counts().to_dict()
    print((sum(k * v for k, v in cnt.items())) / (sum(cnt.values()) * 2))
    cnt = test["cluster_id_cnt"].value_counts().to_dict()
    print((sum(k * v for k, v in cnt.items())) / (sum(cnt.values()) * 2))
