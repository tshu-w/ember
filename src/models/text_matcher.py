from functools import partial
from typing import Any, Optional, Union

import torch.nn as nn
from transformers import AdamW, AutoModel, AutoTokenizer

from .matcher import Matcher


class TextMatcher(Matcher):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: Optional[int] = None,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.convert_to_features = partial(
            self._convert_to_features, tokenizer=tokenizer, max_length=max_length
        )

        self.model = AutoModel.from_pretrained(model_name_or_path)
        dropout = (
            self.model.config.classifier_dropout
            or self.model.config.hidden_dropout_prob
            or dropout
        )
        embed_dim = self.model.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 2))

    def forward(self, inputs) -> Any:
        output = self.model(**inputs).pooler_output
        logits = self.classifier(output)

        return logits

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def _convert_to_features(
        batch: Union[dict[str, list], list[Any]],
        tokenizer,
        max_length: Optional[int] = None,
    ) -> Union[dict, Any]:
        features = tokenizer(
            batch["text_left"],
            batch["text_right"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        features["labels"] = batch["labels"]

        return features
