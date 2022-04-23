from functools import partial
from typing import Any, Optional, Union

import torch.nn as nn
from datasets.features import Array3D, ClassLabel, Features, Sequence, Value
from PIL import Image, ImageFile
from transformers import AdamW, AutoFeatureExtractor, AutoModel, AutoTokenizer

from .matcher import Matcher
from .modules import Fusion, GatedSum

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MultimodalMatcher(Matcher):
    def __init__(
        self,
        text_model_name_or_path: str,
        vision_model_name_or_path: str,
        max_length: Optional[int] = None,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            vision_model_name_or_path
        )
        self.convert_to_features = partial(
            self._convert_to_features,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            max_length=max_length,
        )

        text_features = {
            k: Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None)
            for k in tokenizer.model_input_names
        }
        image_size = feature_extractor.size
        if not isinstance(image_size, tuple):
            image_size = image_size, image_size
        vision_features = {
            "pixel_values_left": Array3D(shape=(None, *image_size), dtype="float32"),
            "pixel_values_right": Array3D(shape=(None, *image_size), dtype="float32"),
        }
        label_features = {
            "labels": ClassLabel(num_classes=2, names=["not_matched", "matched"]),
        }
        self.features = Features(text_features | vision_features | label_features)

        self.text_model = AutoModel.from_pretrained(text_model_name_or_path)
        self.vision_model = AutoModel.from_pretrained(vision_model_name_or_path)

        text_embed_dim = self.text_model.config.hidden_size
        vision_embed_dim = self.vision_model.config.hidden_size
        self.vision_fusion = Fusion(
            input_dim=vision_embed_dim, output_dim=text_embed_dim
        )
        self.gated_sum = GatedSum(text_embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_embed_dim, 2),
        )

    def forward(self, inputs) -> Any:
        pixel_values_left = inputs.pop("pixel_values_left")
        pixel_values_right = inputs.pop("pixel_values_right")

        vision_output_left = self.vision_model(pixel_values_left).pooler_output
        vision_output_right = self.vision_model(pixel_values_right).pooler_output
        vision_output = self.vision_fusion(vision_output_left, vision_output_right)

        text_output = self.text_model(**inputs).pooler_output

        logits = self.classifier(self.gated_sum(vision_output, text_output))
        # logits = self.classifier(torch.cat((vision_output, text_output), dim=1))

        return logits

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def _convert_to_features(
        batch: Union[dict[str, list], list[Any]],
        tokenizer,
        feature_extractor,
        max_length: Optional[int] = None,
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

        image_features_left = feature_extractor(image_left)
        image_features_right = feature_extractor(image_right)

        features = tokenizer(
            batch["text_left"],
            batch["text_right"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        features |= {
            "pixel_values_left": image_features_left["pixel_values"],
            "pixel_values_right": image_features_right["pixel_values"],
            "labels": batch["labels"],
        }

        return features
