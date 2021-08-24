#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .utils import FEATURE_SIZE

ImageFile.LOAD_TRUNCATED_IMAGES = True


class WDCDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        root: Path,
        use_text: bool = True,
        use_image: bool = True,
        feature_type: Optional[str] = None,
        num_image_embeds: int = 1,
        filter_no_image: bool = False,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.dataframe = dataframe
        self.len = len(self.dataframe)

        self.use_text = use_text
        self.use_image = use_image

        self.feature_type = feature_type
        self.num_image_embeds = num_image_embeds

        self.transforms = transforms

        if self.use_image:
            if self.feature_type == "e2e":
                dir = root / "images"
            else:
                dir = root / f"{self.feature_type}_features"

            self.id2paths = defaultdict(list)

            for f in dir.glob("*_*"):
                self.id2paths[int(f.stem[:-2])].append(f)

            for v in self.id2paths.values():
                v.sort()

            if filter_no_image:
                ids = self.id2paths.keys()
                self.dataframe = dataframe[
                    (dataframe["id_left"].isin(ids)) & (dataframe["id_right"].isin(ids))
                ]
                self.len = len(self.dataframe)

    def __getitem__(self, index):
        raw = self.dataframe.iloc[index].to_dict()

        res = {}
        res["raw"] = raw
        res["texts"] = []
        res["images"] = []

        for suffix in ["left", "right"]:
            if self.use_text:
                text = raw[f"title_{suffix}"]
                res["texts"].append(text)

            if self.use_image:
                id = raw[f"id_{suffix}"]

                if self.feature_type == "e2e":
                    image_paths = self.id2paths[id]

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
                    image_paths = self.id2paths[id]

                    if image_paths:
                        image = torch.load(image_paths[0], map_location="cpu")

                        if self.feature_type == "roi":
                            image = image[: self.num_image_embeds, :]
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
        use_text: bool = True,
        use_image: bool = True,
        feature_type: Literal["grid", "roi", "e2e"] = "grid",
        num_image_embeds: int = 1,
        filter_no_image: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cate = cate
        self.training_size = training_size

        self.use_text = use_text
        self.use_image = use_image

        self.feature_type = feature_type
        self.num_image_embeds = num_image_embeds

        self.filter_no_image = filter_no_image

        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def version(self):
        self._version = "_".join(map(str, [self.cate, self.training_size]))

        if self.use_text:
            self._version += "_text"

        if self.use_image:
            self._version += f"_image_{self.feature_type}_{self.num_image_embeds}"

        if self.filter_no_image:
            self._version += f"_filter"

        return self._version

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: Optional[str]) -> None:
        root_dir = Path("../data/wdc/")
        data_dir = Path("../data/wdc/norm/")

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
                use_text=self.use_text,
                use_image=self.use_image,
                feature_type=self.feature_type,
                num_image_embeds=self.num_image_embeds,
                filter_no_image=self.filter_no_image,
                transforms=self.transforms,
            )
            self.data_valid = WDCDataset(
                dataframe=training_df[training_df["pair_id"].isin(validation_pair_id)],
                root=root_dir,
                use_text=self.use_text,
                use_image=self.use_image,
                feature_type=self.feature_type,
                num_image_embeds=self.num_image_embeds,
                filter_no_image=self.filter_no_image,
                transforms=self.transforms,
            )

        if stage == "test" or stage is None:
            test_path = data_dir / "gold-standards" / f"{self.cate}_gs.json.gz"

            self.data_test = WDCDataset(
                dataframe=pd.read_json(test_path, lines=True),
                root=root_dir,
                use_text=self.use_text,
                use_image=self.use_image,
                feature_type=self.feature_type,
                num_image_embeds=self.num_image_embeds,
                filter_no_image=self.filter_no_image,
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
            pin_memory=True,
            collate_fn=self.collate_fn,
            prefetch_factor=4,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
            prefetch_factor=4,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
            prefetch_factor=4,
        )
