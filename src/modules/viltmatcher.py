#!/usr/bin/env python
# -*- coding: utf-8 -*-


from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import F1, MetricCollection, Precision, Recall
from torchvision import transforms
from transformers import AdamW, AutoModel, AutoTokenizer, PreTrainedTokenizer

from .vilt import ViLT


class MinMaxResize:
    def __init__(self, shorter=800, longer=1333):
        self.min = shorter
        self.max = longer

    def __call__(self, x):
        w, h = x.size
        scale = self.min / min(w, h)
        if h < w:
            newh, neww = self.min, scale * w
        else:
            newh, neww = scale * h, self.min

        if max(newh, neww) > self.max:
            scale = self.max / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        newh, neww = int(newh + 0.5), int(neww + 0.5)
        newh, neww = newh // 32 * 32, neww // 32 * 32

        return x.resize((neww, newh), resample=Image.BICUBIC)


def get_transforms(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            # transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),  # TODO: replace padding with simply crop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def collate_fn(
    batch, tokenizer: PreTrainedTokenizer, max_text_length: Optional[int] = 256,
):
    texts = [x["texts"] for x in batch]
    images = [x["images"] for x in batch]

    raws = [x["raw"] for x in batch]
    labels = torch.LongTensor([x["label"] for x in raws])

    text1, text2 = map(list, zip(*texts))
    text1 = tokenizer(
        text1,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    text2 = tokenizer(
        text2,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )

    if images[0]:
        image1 = torch.stack([imgs[0] for imgs in images])
        image2 = torch.stack([imgs[1] for imgs in images])
    else:
        image1 = torch.Tensor(len(batch), 0)
        image2 = torch.Tensor(len(batch), 0)

    inputs = {"text1": text1, "text2": text2, "image1": image1, "image2": image2}

    return inputs, labels, raws


class ViLTMatcher(LightningModule):
    def __init__(
        self,
        model_name: str = "vilt_200k_mlm_itm.ckpt",
        tokenizer_name: str = "bert-base-uncased",
        lr: float = 2e-05,
        max_text_length: int = 40,
        max_image_length: int = -1,
        image_size: int = 384,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if "vilt" in model_name:
            self.model = ViLT.from_pretrained(
                model_name,
                max_text_length=max_text_length,
                max_image_length=max_image_length,
            )
            num_features = self.model.vit.num_features
        else:
            self.model = AutoModel.from_pretrained(model_name)
            num_features = self.model.config.hidden_size
            tokenizer_name = model_name

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features * 4, 2)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collate_fn = partial(
            collate_fn, tokenizer=tokenizer, max_text_length=max_text_length
        )
        self.transforms = get_transforms(size=image_size)

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
        text1, text2, = x["text1"], x["text2"]
        image1, image2 = x["image1"], x["image2"]

        if isinstance(self.model, ViLT):
            out1 = self.model(text1, image1)
            out2 = self.model(text2, image2)
            masks = torch.cat((out1.masks, out2.masks), dim=1)
        else:
            out1 = self.model(**text1)
            out2 = self.model(**text2)
            masks = torch.cat((text1.attention_mask, text2.attention_mask), dim=1)

        hidden_state = torch.cat(
            (out1.last_hidden_state, out2.last_hidden_state), dim=1
        )
        out = self.transformer(hidden_state, src_key_padding_mask=masks)
        cls1 = out[:, 0]
        cls2 = out[:, out1.last_hidden_state.shape[1]]

        # cls1 = out1.pooler_output
        # cls2 = out2.pooler_output
        cls = torch.cat((cls1, cls2, cls1 - cls2, cls1 * cls2), dim=1)
        cls = self.dropout(cls)

        logits = self.classifier(cls)

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
