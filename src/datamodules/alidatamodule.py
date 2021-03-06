import warnings
from functools import partial
from pathlib import Path
from typing import Literal, Optional

from datasets.features.features import Features
from datasets.load import load_dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)


class AliDataModule(LightningDataModule):
    def __init__(
        self,
        cat: Literal["all", "clothing", "shoes", "accessories"] = "all",
        columns: list[str] = ["title", "pv_pairs"],
        test_name: str = "",
        train_ratio: Optional[int] = None,
        test_ratio: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cat = cat
        self.columns = columns
        self.data_path = Path("./data/ali/")
        self.image_path = self.data_path / "images"
        self.train_path = self.data_path / f"datasets/{cat}/train.parquet"
        self.val_path = self.data_path / f"datasets/{cat}/val.parquet"
        self.test_name = test_name
        test_file_name = f"test_{test_name}" if test_name else "test"
        self.test_path = self.data_path / f"datasets/{cat}/{test_file_name}.parquet"

        if train_ratio is not None:
            self.train_path = (
                self.data_path / f"datasets_ratio/{cat}/train_{train_ratio}.parquet"
            )
            self.val_path = (
                self.data_path / f"datasets_ratio/{cat}/val_{train_ratio}.parquet"
            )
        if test_ratio is not None:
            self.test_path = (
                self.data_path
                / f"datasets_ratio/{cat}/{test_file_name}_{test_ratio}.parquet"
            )

    def prepare_data(self) -> None:
        self.setup()  # avoid cache conflict in multi processes

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "datasets"):
            convert_to_features = self.trainer.model.convert_to_features
            features = getattr(self.trainer.model, "features", None)
            preprocess_fn = partial(
                self._preprocess, columns=self.columns, image_path=self.image_path
            )
            preprocess = lambda x: convert_to_features(preprocess_fn(x))

            data_files = {
                "train": str(self.train_path),
                "val": str(self.val_path),
                "test": str(self.test_path),
            }

            datasets = load_dataset("parquet", data_files=data_files)
            self.datasets = datasets.map(
                preprocess,
                batched=True,
                remove_columns=next(iter(datasets.values())).column_names,
                features=Features(features) if features else None,
            )
            self.datasets.set_format(type="torch")

        self.collate_fn = getattr(self.trainer.model, "collate_fn", None)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["val"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0,
            shuffle=False,
        )

    @staticmethod
    def _preprocess(batch, columns: list[str], image_path: Path):
        batch["pv_pairs_left"] = [
            pv_pairs.replace("#;#", " ").replace("#:#", " ")
            for pv_pairs in batch["pv_pairs_left"]
        ]
        batch["pv_pairs_right"] = [
            pv_pairs.replace("#;#", " ").replace("#:#", " ")
            for pv_pairs in batch["pv_pairs_right"]
        ]

        text_left = []
        for attrs in zip(*(batch[f"{c}_left"] for c in columns)):
            text_left.append(" ".join(map(lambda x: str(x or ""), attrs)))

        text_right = []
        for attrs in zip(*(batch[f"{c}_right"] for c in columns)):
            text_right.append(" ".join(map(lambda x: str(x or ""), attrs)))

        image_left = [image_path / f"{i}.jpg" for i in batch["id_left"]]
        image_right = [image_path / f"{i}.jpg" for i in batch["id_right"]]

        return {
            "text_left": text_left,
            "text_right": text_right,
            "image_left": image_left,
            "image_right": image_right,
            "labels": batch["label"],
        }
