#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils import FEATURE_SIZE

ImageFile.LOAD_TRUNCATED_IMAGES = True

class WDCDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        root: Path,
        use_image: bool = False,
        feature_type: Optional[str] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.dataframe = dataframe
        self.root = root
        self.len = len(dataframe)

        self.use_image = use_image
        self.feature_type = feature_type
        self.transforms = transforms

    def __getitem__(self, index):
        raw = self.dataframe.iloc[index].to_dict()

        res = {}
        res["raw"] = raw
        res["texts"] = []
        res["images"] = []

        for suffix in ["left", "right"]:
            text = raw[f"title_{suffix}"]

            res["texts"].append(text)

            if self.use_image:
                id = raw[f"id_{suffix}"]
                if self.feature_type == "e2e":
                    image_paths = sorted(self.root.glob(f"images/{id}_*"))

                    if image_paths:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            image = Image.open(image_paths[0]).convert("RGB")

                        image = self.transforms(image)
                    else:
                        image = Image.fromarray(
                            255 * np.ones((256, 256, 3), dtype=np.uint8)
                        )
                        image = torch.zeros_like(self.transforms(image))
                else:
                    image_paths = sorted(
                        self.root.glob(f"{self.feature_type}_features/{id}_*")
                    )

                    if image_paths:
                        image = torch.load(image_paths[0], map_location="cpu")
                    else:
                        if self.feature_type == "grid":
                            image = torch.zeros(*FEATURE_SIZE[self.feature_type])
                        else:
                            image = torch.zeros(0, *FEATURE_SIZE[self.feature_type])

                res["images"].append(image)

        return res

    def __len__(self):
        return self.len


class WDCDataModule(LightningDataModule):
    def __init__(
        self,
        cate: Literal["all", "cameras", "computers", "shoes", "watches"] = "all",
        training_size: Literal["small", "medium", "large", "xlarge"] = "medium",
        use_image: bool = True,
        extended: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cate = cate
        self.training_size = training_size
        self.use_image = use_image

        self.extended = extended

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.version = f"{cate}_{training_size}_{use_image}_{extended}"

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: Optional[str]) -> None:
        data_dir = Path("../data/wdc/norm/")
        root_dir = Path("../data/wdc/")

        if stage in ["fit", "validate"] or stage is None:
            training_path = (
                data_dir
                / "training-sets"
                / f"{self.cate}_train"
                / f"{self.cate}_train_{self.training_size}.json.gz"
            )
            training_df = pd.read_json(training_path, lines=True)

            validation_set_path = (
                data_dir
                / "validation-sets"
                / f"{self.cate}_valid"
                / f"{self.cate}_valid_{self.training_size}.csv"
            )
            validation_pair_id = pd.read_csv(validation_set_path)["pair_id"]

            self.data_train = WDCDataset(
                dataframe=training_df[~training_df["pair_id"].isin(validation_pair_id)],
                root=root_dir,
                use_image=self.use_image,
                feature_type=self.feature_type,
                transforms=self.transforms,
            )
            self.data_valid = WDCDataset(
                dataframe=training_df[training_df["pair_id"].isin(validation_pair_id)],
                root=root_dir,
                use_image=self.use_image,
                feature_type=self.feature_type,
                transforms=self.transforms,
            )

        if stage == "test" or stage is None:
            if not self.extended:
                test_path = data_dir / "gold-standards" / f"{self.cate}_gs.json.gz"
            else:
                test_path = data_dir / "test-sets" / f"{self.cate}_test.json.gz"
                if not test_path.exists():
                    from .extended_wdc import main as extended_wdc

                    extended_wdc()

            self.data_test = WDCDataset(
                dataframe=pd.read_json(test_path, lines=True),
                root=root_dir,
                use_image=self.use_image,
                feature_type=self.feature_type,
                transforms=self.transforms,
            )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )
