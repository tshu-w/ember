#!/usr/bin/env python

from functools import partial
from itertools import chain
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torchmetrics import F1, MetricCollection, Precision, Recall

from src.utils import FEATURE_SIZE

from .mmtsmatcher import ImageEncoder, get_transforms


def collate_fn(
    batch,
    num_image_embeds: int = 1,
):
    batch_size = len(batch)
    images = [x["images"] for x in batch]

    raws = [x["raw"] for x in batch]
    labels = torch.LongTensor([x["label"] for x in raws])

    images = list(chain.from_iterable(zip(*images)))
    images_tensor = pad_sequence(images, batch_first=True)

    if len(images_tensor.shape) == 5:  # 2BxNxDx7x7
        images_tensor = images_tensor[:, :num_image_embeds, :]

    images = torch.stack(
        [images_tensor[:batch_size, :], images_tensor[batch_size:, :]], dim=1
    )

    return images, labels, raws


class ImageMatcher(LightningModule):
    def __init__(
        self,
        feature_type: Literal["grid", "roi", "e2e"] = "grid",
        lr: float = 1e-04,
        num_image_embeds: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ImageEncoder(num_image_embeds, feature_type)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            num_image_embeds * FEATURE_SIZE[feature_type][0] * 4, 2
        )

        self.collate_fn = partial(collate_fn, num_image_embeds=num_image_embeds)
        self.transforms = get_transforms()

        self.feature_type = feature_type
        self.num_image_embeds = num_image_embeds

        self.lr = lr

        metrics_kwargs = {"ignore_index": 0}
        metrics = MetricCollection(
            {
                "f1": F1(**metrics_kwargs),
                "prc": Precision(**metrics_kwargs),
                "rec": Recall(**metrics_kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x, labels):
        image1, image2 = x[:, 0], x[:, 1]
        image1 = self.encoder(image1).flatten(start_dim=1)
        image2 = self.encoder(image2).flatten(start_dim=1)

        combined = torch.cat([image1, image2, image1 - image2, image1 * image2], dim=1)
        logits = self.classifier(self.dropout(combined))

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return logits, loss

    def common_step(self, batch, step: str):
        x, labels, _ = batch
        logits, loss = self.forward(x, labels)
        probs = F.softmax(logits, dim=-1)

        metrics = getattr(self, f"{step}_metrics")
        metrics(probs, labels)

        self.log_dict(metrics, prog_bar=True)
        self.log(f"{step}_loss", loss, prog_bar=True)

        return loss

    def training_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> STEP_OUTPUT:
        return self.common_step(batch, "train")

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Optional[STEP_OUTPUT]:
        return self.common_step(batch, "valid")

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Optional[STEP_OUTPUT]:
        return self.common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        if "v_num" in items:
            items.pop("v_num")
        return items

    def configure_callbacks(self):
        callbacks_args = {"monitor": "valid_f1", "mode": "max"}

        early_stop = EarlyStopping(patience=5, **callbacks_args)
        checkpoint = ModelCheckpoint(
            filename="{epoch:02d}-{valid_f1:.2%}", **callbacks_args
        )

        return [early_stop, checkpoint]
