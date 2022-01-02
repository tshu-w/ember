#!/usr/bin/env python

from functools import partial
from typing import Any, Union

import torch.nn as nn
from datasets.features import Array3D, ClassLabel, Features
from PIL import Image, ImageFile
from transformers import AdamW, AutoFeatureExtractor, AutoModel

from .matcher import Matcher
from .modules import Fusion

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VisionMatcher(Matcher):
    def __init__(
        self,
        model_name_or_path: str,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.convert_to_features = partial(
            self._convert_to_features, feature_extractor=feature_extractor
        )

        image_size = feature_extractor.size
        if not isinstance(image_size, tuple):
            image_size = image_size, image_size
        self.features = Features(
            {
                "pixel_values_left": Array3D(
                    shape=(None, *image_size), dtype="float32"
                ),
                "pixel_values_right": Array3D(
                    shape=(None, *image_size), dtype="float32"
                ),
                "labels": ClassLabel(num_classes=2, names=["not_matched", "matched"]),
            }
        )

        self.model = AutoModel.from_pretrained(model_name_or_path)
        embed_dim = self.model.config.hidden_size
        self.fusion = Fusion(input_dim=embed_dim, output_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, inputs) -> Any:
        output_left = self.model(inputs["pixel_values_left"]).pooler_output
        output_right = self.model(inputs["pixel_values_right"]).pooler_output
        pooled_output = self.fusion(output_left, output_right)
        logits = self.classifier(pooled_output)

        return logits

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def get_version(self):
        return self.hparams.model_name_or_path

    @staticmethod
    def _convert_to_features(
        batch: Union[dict[str, list], list[Any]],
        feature_extractor,
    ) -> Union[dict, Any]:
        blank_image = Image.new("RGB", (256, 256), (255, 255, 255))

        image_left = [
            Image.open(i).convert("RGB") if i else blank_image
            for i in batch["image_left"]
        ]
        image_right = [
            Image.open(i).convert("RGB") if i else blank_image
            for i in batch["image_right"]
        ]

        vision_features_left = feature_extractor(image_left)
        vision_features_right = feature_extractor(image_right)

        features = {
            "pixel_values_left": vision_features_left["pixel_values"],
            "pixel_values_right": vision_features_right["pixel_values"],
            "labels": batch["labels"],
        }

        return features
