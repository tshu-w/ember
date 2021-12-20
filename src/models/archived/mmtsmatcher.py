#!/usr/bin/env python

import json
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
import torchvision
from numpyencoder import NumpyEncoder
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import F1, MetricCollection, Precision, Recall
from torchvision import transforms
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from src.utils import FEATURE_SIZE

from .mmts import MMTSConfig, MMTSForSequenceClassification


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collate_fn(
    batch,
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    num_image_embeds: int = 1,
):
    batch_size = len(batch)
    texts = [x["texts"] for x in batch]
    images = [x["images"] for x in batch]

    raws = [x["raw"] for x in batch]
    labels = torch.LongTensor([x["label"] for x in raws])

    sent1, sent2 = map(list, zip(*texts))

    if max_length:
        max_length = max_length - 2 * num_image_embeds - 1

    inputs = tokenizer(
        sent1,
        sent2,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    if images[0]:
        images = list(chain.from_iterable(zip(*images)))
        images_tensor = pad_sequence(images, batch_first=True)

        if len(images_tensor.shape) == 5:  # 2BxNxDx7x7
            images_tensor = images_tensor[:, :num_image_embeds, :]

        inputs["input_modals"] = torch.stack(
            [images_tensor[:batch_size, :], images_tensor[batch_size:, :]], dim=1
        )
    else:
        inputs["input_modals"] = torch.Tensor(batch_size, 0)

    return inputs, labels, raws


class ImageEncoder(nn.Module):
    def __init__(self, num_image_embeds, feature_type):
        super().__init__()

        self.feature_type = feature_type

        if feature_type == "e2e":
            model = torchvision.models.resnet152(pretrained=True)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
        else:
            self.model = nn.Identity()

        if feature_type == "roi":
            self.num_image_embeds = num_image_embeds
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            POOLING_BREAKDOWN = {
                1: (1, 1),
                2: (2, 1),
                3: (3, 1),
                4: (2, 2),
                5: (5, 1),
                6: (3, 2),
                7: (7, 1),
                8: (4, 2),
                9: (3, 3),
            }
            self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[num_image_embeds])

    def forward(self, x):
        # grid: (Bx3x224x224) -> BxDx7x7 -> BxDxN -> BxNxD
        # roi:  BxNxDx7x7 -> BxNxDx1x1 -> BxNxD
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)

        if self.feature_type == "roi":
            out = out[:, : self.num_image_embeds, :]
        else:
            out = out.transpose(1, 2).contiguous()

        return out  # BxNxD


class MMTSMatcher(LightningModule):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        feature_type: Literal["grid", "roi", "e2e"] = "grid",
        lr: float = 1e-05,
        max_length: int = 256,
        num_image_embeds: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(model_name)
        config = MMTSConfig(
            config, num_labels=2, modal_hidden_size=FEATURE_SIZE[feature_type][0]
        )
        if "visualbert" not in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        transformers = AutoModel.from_pretrained(model_name)
        image_encoder = ImageEncoder(num_image_embeds, feature_type)
        self.model = MMTSForSequenceClassification(
            config, transformers, image_encoder, tokenizer.sep_token_id
        )

        self.collate_fn = partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            num_image_embeds=num_image_embeds,
        )

        self.feature_type = feature_type
        self.num_image_embeds = num_image_embeds
        self.transforms = get_transforms()

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
        y = self.model(**x, labels=labels, return_dict=True)
        return y

    def common_step(self, batch, step: str):
        x, labels, row = batch
        y = self.forward(x, labels)
        probs = F.softmax(y.logits, dim=-1)

        metrics = getattr(self, f"{step}_metrics")
        metrics(probs, labels)

        self.log_dict(metrics, prog_bar=True)
        self.log(f"{step}_loss", y.loss, prog_bar=True)

        if step == "test":
            preds = probs.argmax(dim=-1)
            errors = torch.nonzero(torch.ne(preds, labels))
            error_cases = [
                json.dumps(row[i], ensure_ascii=False, indent=2, cls=NumpyEncoder)
                + "\n"
                for i in errors.squeeze(dim=-1).tolist()
            ]
            errors_cases_file = Path(self.trainer.log_dir) / f"error_cases_{step}.json"
            with errors_cases_file.open("a") as f:
                f.writelines(error_cases)

        return y.loss

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
