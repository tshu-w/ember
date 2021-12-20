#!/usr/bin/env python

import json
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpyencoder import NumpyEncoder
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import F1, MetricCollection, Precision, Recall
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .mmts import MMTSConfig, MMTSModel
from .modules import GatedSum, ImageEncoder, ImageMatcher, TextMatcher
from .utils import FEATURE_SIZE, collate_fn, get_transforms


class MultimodalMatcher(LightningModule):
    def __init__(
        self,
        text_image: str = Literal["aligned", "seperated"],
        text_text: str = Literal["cross", "dual"],
        use_text: bool = True,
        use_image: bool = True,
        model_name: str = "bert-base-uncased",
        max_length: int = 256,
        feature_type: Literal["grid", "roi", "e2e"] = "grid",
        num_image_embeds: int = 1,
        dropout: float = 0.1,
        lr: Optional[float] = None,
    ):
        super().__init__()

        self.text_image = text_image
        self.text_text = text_text
        self.use_text = use_text
        self.use_image = use_image

        self.model_name = model_name

        assert self.use_text or self.use_image

        if self.text_image == "aligned":
            # assert self.use_text and self.use_image

            config = AutoConfig.from_pretrained(model_name)
            config = MMTSConfig(config, modal_hidden_size=FEATURE_SIZE[feature_type][0])

            if "visualbert" not in model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            transformers = AutoModel.from_pretrained(model_name)
            image_encoder = ImageEncoder(
                feature_type=feature_type, num_image_embeds=num_image_embeds
            )
            model = MMTSModel(
                config=config,
                transformer=transformers,
                encoder=image_encoder,
                sep_token_id=tokenizer.sep_token_id,
            )

            self.model = TextMatcher(model=model, type=self.text_text)

            if hasattr(model.config, "hidden_dropout_prob"):
                dropout = model.config.hidden_dropout_prob
            output_dim = self.model.get_output_dim()

        elif self.text_image == "seperated":
            if self.use_text:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)

                self.text_model = TextMatcher(model=model, type=self.text_text)

                if hasattr(model.config, "hidden_dropout_prob"):
                    dropout = model.config.hidden_dropout_prob
                output_dim = self.text_model.get_output_dim()
            else:
                tokenizer = None

            if self.use_image:
                model = ImageEncoder(
                    feature_type=feature_type, num_image_embeds=num_image_embeds
                )
                self.image_model = ImageMatcher(
                    model=model,
                    input_dim=FEATURE_SIZE[feature_type][0] * num_image_embeds,
                    output_dim=output_dim if self.use_text else None,
                )
                output_dim = self.image_model.get_output_dim()

            if self.use_text and self.use_image:
                self.gated_sum = GatedSum(input_dim=output_dim)

        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(output_dim, 2))
        self.loss = nn.CrossEntropyLoss()

        self.collate_fn = partial(
            collate_fn,
            text_image=text_image,
            text_text=text_text,
            use_text=use_text,
            use_image=use_image,
            tokenizer=tokenizer,
            max_length=max_length,
            num_image_embeds=num_image_embeds,
        )
        if self.use_image and feature_type == "e2e":
            self.transforms = get_transforms()
        else:
            self.transforms = None

        if lr:
            self.lr = lr
        else:
            if self.use_text:
                self.lr = 1e-05
            else:
                self.lr = 1e-04

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

    def forward(self, inputs, labels):
        if self.text_image == "aligned":
            output = self.model(inputs)
        elif self.text_image == "seperated":
            texts, images = inputs
            if self.use_text:
                text_output = self.text_model(texts)
            if self.use_image:
                image_output = self.image_model(images)

            if self.use_text and self.use_image:
                output = self.gated_sum(text_output, image_output)
            else:
                output = text_output if self.use_text else image_output

        logits = self.classifier(output)

        if labels is not None:
            loss = self.loss(logits, labels)

        return logits, loss

    def common_step(self, batch, step: str):
        *inputs, labels, row = batch
        logits, loss = self.forward(*inputs, labels)
        probs = F.softmax(logits, dim=-1)

        metrics = getattr(self, f"{step}_metrics")
        metrics(probs, labels)

        self.log_dict(metrics, prog_bar=True)
        self.log(f"{step}_loss", loss, prog_bar=True)

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

        return loss

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        return self.common_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self.common_step(batch, "valid")

    def test_step(self, batch, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self.common_step(batch, "test")

    def configure_optimizers(self):
        if self.use_text:
            from transformers import AdamW
        else:
            from torch.optim import AdamW

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

    def get_version(self):
        self._version = f"{self.text_image}_{self.text_text}"
        if self.use_text:
            self._version += f"_{self.model_name}"

        return self._version
