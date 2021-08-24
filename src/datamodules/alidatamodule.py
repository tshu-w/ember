#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .utils import ALI_CATE_LEVEL_NAME, ALI_CATE_NAME, FEATURE_SIZE, train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ALIDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        root: Path,
        use_text: bool = True,
        use_image: bool = True,
        use_pv_pairs: bool = False,
        feature_type: Optional[str] = None,
        num_image_embeds: int = 1,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.dataframe = dataframe
        self.len = len(dataframe)

        self.use_text = use_text
        self.use_image = use_image
        self.use_pv_pairs = use_pv_pairs

        self.feature_type = feature_type
        self.num_image_embeds = num_image_embeds
        self.transforms = transforms

        if self.feature_type == "e2e":
            self.image_dir = root / "images"
            images = self.image_dir.glob("*.jpg")
        else:
            self.image_dir = root / f"{self.feature_type}_features"
            images = self.image_dir.glob("*.pt")

        self.ids = set([int(f.stem) for f in images])

    def __getitem__(self, index: int):
        raw = self.dataframe.iloc[index].to_dict()

        res = {}
        res["raw"] = raw
        res["texts"] = []
        res["images"] = []

        def serialize_pv_pairs(pv_pairs):
            return " ".join([" ".join(p.split("#:#")) for p in pv_pairs.split("#;#")])

        for suffix in ["left", "right"]:
            if self.use_text:
                text = raw[f"title_{suffix}"]

                if self.use_pv_pairs:
                    pv_pairs = serialize_pv_pairs(raw[f"pv_pairs_{suffix}"])
                    text += " " + pv_pairs

                res["texts"].append(text)

            if self.use_image:
                id = raw[f"id_{suffix}"]

                if self.feature_type == "e2e":
                    image_path = self.image_dir / f"{id}.jpg"

                    if id in self.ids:
                        image = Image.open(image_path).convert("RGB")
                        image = self.transforms(image)
                    else:
                        image = Image.fromarray(
                            255 * np.ones((256, 256, 3), dtype=np.uint8)
                        )
                        image = torch.zeros_like(self.transforms(image))
                else:
                    image_path = self.image_dir / f"{id}.pt"

                    if id in self.ids:
                        image = torch.load(image_path, map_location="cpu")

                        if self.feature_type == "roi":
                            image = image[: self.num_image_embeds, :]
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
        use_text: bool = True,
        use_image: bool = True,
        use_pv_pairs: bool = False,
        feature_type: Literal["grid", "roi", "e2e"] = "grid",
        num_image_embeds: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cate_level_name = cate_level_name
        self.cate_name = cate_name
        self.prod_num = prod_num

        self.use_text = use_text
        self.use_image = use_image
        self.use_pv_pairs = use_text and use_pv_pairs

        self.feature_type = feature_type
        self.num_image_embeds = num_image_embeds

        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def version(self):
        self._version = "_".join(
            map(str, [self.cate_level_name, self.cate_name, self.prod_num])
        )

        if self.use_text:
            self._version += f"_text_{self.use_pv_pairs}"

        if self.use_image:
            self._version += f"_image_{self.feature_type}_{self.num_image_embeds}"

        return self._version

    def prepare_data(self) -> None:
        cate_level_name = (
            ("_" + self.cate_level_name.replace("/", "_"))
            if self.cate_level_name
            else ""
        )
        cate_name = ("_" + self.cate_name.replace("/", "_")) if self.cate_name else ""

        self.data_path = Path(
            f"../data/ali/dataset/dataset{cate_level_name}{cate_name}_{self.prod_num}.json"
        )
        self.test_path = Path(
            f"../data/ali/testset/testset{cate_level_name}{cate_name}_{self.prod_num}.json"
        )
        self.root = Path(f"../data/ali")

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit" or stage is None:
            dataframe = pd.read_json(self.data_path, lines=True)

            dataset = ALIDataset(
                dataframe=dataframe,
                root=self.root,
                use_text=self.use_text,
                use_image=self.use_image,
                use_pv_pairs=self.use_pv_pairs,
                feature_type=self.feature_type,
                num_image_embeds=self.num_image_embeds,
                transforms=self.transforms,
            )
            self.data_train, self.data_valid = train_test_split(dataset, test_size=0.2)

        if stage == "test" or stage is None:
            dataframe = pd.read_json(self.test_path, lines=True)

            self.data_test = ALIDataset(
                dataframe=dataframe,
                root=self.root,
                use_text=self.use_text,
                use_image=self.use_image,
                use_pv_pairs=self.use_pv_pairs,
                feature_type=self.feature_type,
                num_image_embeds=self.num_image_embeds,
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
