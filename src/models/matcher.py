from typing import Optional

import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import F1Score, MetricCollection, Precision, Recall


class Matcher(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        metrics_kwargs = {"ignore_index": 0}
        metrics = MetricCollection(
            {
                "f1": F1Score(**metrics_kwargs),
                "prc": Precision(**metrics_kwargs),
                "rec": Recall(**metrics_kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")

    def common_step(self, batch, step: str) -> Optional[STEP_OUTPUT]:
        labels = batch.pop("labels", None)
        logits = self.forward(batch)

        if labels is not None:
            loss = F.cross_entropy(logits, labels) if labels is not None else None

            metrics = getattr(self, f"{step}_metrics")
            probs = F.softmax(logits, dim=-1)
            metrics(probs, labels)

            self.log_dict(metrics, prog_bar=True)
            self.log(f"{step}_loss", loss, prog_bar=True)
        else:
            loss = None

        return loss

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        return self.common_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self.common_step(batch, "valid")

    def test_step(self, batch, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self.common_step(batch, "test")

    def configure_callbacks(self):
        callbacks_kargs = {"monitor": "valid_f1", "mode": "max"}

        early_stop = EarlyStopping(patience=5, **callbacks_kargs)
        checkpoint = ModelCheckpoint(
            filename="{epoch:02d}-{valid_f1:.2%}", **callbacks_kargs
        )

        return [early_stop, checkpoint]
