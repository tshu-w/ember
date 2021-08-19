#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import linecache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .build_dataset import build_dataset
from src.utils import ALI_CATE_LEVEL_NAME, ALI_CATE_NAME, FEATURE_SIZE, train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ALIDataset(Dataset):
    def __init__(
        self,
        filename: Union[str, Path],
        dataframe: pd.DataFrame,
        use_image: bool = False,
        use_pv_pairs: bool = False,
        feature_type: Optional[str] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.dataframe = dataframe
        self.filename = filename
        self.len = len(dataframe)

        self.use_image = use_image
        self.use_pv_pairs = use_pv_pairs

        self.feature_type = feature_type
        self.transforms = transforms

    def __getitem__(self, index: int):
        raw = self.dataframe.iloc[index].to_dict()

        res = {}
        res["raw"] = raw
        res["texts"] = []
        res["images"] = []

        def serialize_pv_pairs(pv_pairs):
            return " ".join([" ".join(p.split("#:#")) for p in pv_pairs.split("#;#")])

        root = Path(self.filename).parent

        for suffix in ["left", "right"]:
            text = raw[f"title_{suffix}"]

            if self.use_pv_pairs:
                pv_pairs = serialize_pv_pairs(raw[f"pv_pairs_{suffix}"])
                text += " " + pv_pairs

            res["texts"].append(text)

            if self.use_image:
                if self.feature_type == "e2e":
                    image_path = root / "images" / (str(raw[f"id_{suffix}"]) + ".jpg")
                    if image_path.exists():
                        image = Image.open(image_path).convert("RGB")
                        image = self.transforms(image)
                    else:
                        image = Image.fromarray(
                            255 * np.ones((256, 256, 3), dtype=np.uint8)
                        )
                        image = torch.zeros_like(self.transforms(image))
                else:
                    image_path = (
                        root
                        / (str(self.feature_type) + "_features")
                        / (str(raw[f"id_{suffix}"]) + ".pt")
                    )

                    if image_path.exists():
                        image = torch.load(image_path, map_location="cpu")
                    else:
                        if self.feature_type == "grid":
                            image = torch.zeros(*FEATURE_SIZE[self.feature_type])
                        else:
                            image = torch.zeros(0, *FEATURE_SIZE[self.feature_type])

                res["images"].append(image)

        return res

    def __len__(self) -> int:
        return self.len


class AliDataModule(LightningDataModule):
    def __init__(
        self,
        cate_level_name: Optional[ALI_CATE_LEVEL_NAME] = None,
        cate_name: Optional[ALI_CATE_NAME] = None,
        prod_num: int = 200,
        use_image: bool = False,
        use_pv_pairs: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cate_level_name = cate_level_name
        self.cate_name = cate_name
        self.prod_num = prod_num

        self.use_image = use_image
        self.use_pv_pairs = use_pv_pairs

        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def version(self):
        self._version = "_".join(
            map(
                str,
                [
                    self.cate_level_name,
                    self.cate_name,
                    self.prod_num,
                    self.use_pv_pairs,
                ],
            )
        )

        if self.use_image:
            self._version += f"_{self.feature_type}_{self.num_image_embeds}"
        else:
            self._version += f"_text"

        return self._version

    def prepare_data(self) -> None:
        column_names = [
            "id",
            "title",
            "pict_url",
            "cate_name",
            "cate_level_name",
            "pv_pairs",
            "cluster_id",
        ]

        cate_level_name = (
            ("_" + self.cate_level_name.replace("/", "_"))
            if self.cate_level_name
            else ""
        )
        cate_name = ("_" + self.cate_name.replace("/", "_")) if self.cate_name else ""

        self.data_path = Path(
            f"../data/ali/dataset{cate_level_name}{cate_name}_{self.prod_num}.json"
        )
        self.test_path = Path(
            f"../data/ali/testset{cate_level_name}{cate_name}_{self.prod_num}.json"
        )

        if not self.data_path.exists() or not self.test_path.exists():
            df = pd.read_csv(
                "../data/ali/same_product_train_sample_1wpid_USTC.txt",
                header=None,
                sep="@;@",
                names=column_names,
                engine="python",
            )

            if not self.data_path.exists():
                build_dataset(
                    df,
                    cate_name=self.cate_name,
                    cate_level_name=self.cate_level_name,
                    num=self.prod_num,
                    path=self.data_path,
                )

            if not self.test_path.exists():
                build_dataset(
                    df,
                    cate_name=self.cate_name,
                    cate_level_name=self.cate_level_name,
                    num=self.prod_num,
                    path=self.test_path,
                    size=5000,
                )

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit" or stage is None:
            dataframe = pd.read_json(self.data_path, lines=True)

            dataset = ALIDataset(
                dataframe=dataframe,
                filename=self.data_path,
                use_image=self.use_image,
                use_pv_pairs=self.use_pv_pairs,
                feature_type=self.feature_type,
                transforms=self.transforms,
            )
            self.data_train, self.data_valid = train_test_split(dataset, test_size=0.2)

        if stage == "test" or stage is None:
            dataframe = pd.read_json(self.test_path, line=True)

            self.data_test = ALIDataset(
                dataframe=dataframe,
                filename=self.filename,
                use_image=self.use_image,
                use_pv_pairs=self.use_pv_pairs,
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
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
