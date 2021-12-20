#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import vilt.modules.vision_transformer as vit
from transformers import AutoConfig, AutoModel
from transformers.file_utils import ModelOutput

# class ViLTConfig(object):
#     config = {
#         "name":  "vit_base_patch32_384",
#         "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth",
#         "num_classes": 1000,
#         "pool_size": None,
#         "interpolation": "bicubic",
#         # "mean": IMAGENET_DEFAULT_MEAN,
#         # "std": IMAGENET_DEFAULT_STD,
#         "first_conv": "patch_embed.proj",
#         "classifier": "head",
#         "input_size": (3, 384, 384),
#         "mean": (0.5, 0.5, 0.5),
#         "std": (0.5, 0.5, 0.5),
#         "crop_pct": 1.0,
#     }


@dataclass
class ViLTModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    masks: torch.FloatTensor = None
    text_hidden_state: torch.FloatTensor = None
    image_hidden_state: torch.FloatTensor = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViLT(nn.Module):
    def __init__(
        self,
        model_type: str = "bert",
        vit_name: str = "vit_base_patch32_384",
        max_text_length: int = 40,
        max_image_length: int = -1,
    ):
        super().__init__()

        config = AutoConfig.for_model(model_type)
        config.max_position_embeddings = max_text_length
        model = AutoModel.from_config(config)

        self.text_embeddings = model.embeddings
        self.token_type_embeddings = deepcopy(
            self.text_embeddings.token_type_embeddings
        )

        self.vit = getattr(vit, vit_name)()
        self.pooler = Pooler(config.hidden_size)

        self.max_text_length = max_text_length
        self.max_image_length = max_image_length

    def forward(self, text, image):
        text_input_ids, text_masks = text.input_ids, text.attention_mask

        text_embeds = self.text_embeddings(text_input_ids)
        text_embeds += self.token_type_embeddings(torch.zeros_like(text_masks))

        if image.size(1):
            (
                image_embeds,
                image_masks,
                _patch_index,
                _image_label,
            ) = self.vit.visual_embed(image, max_image_len=self.max_image_length)
            image_embeds += self.token_type_embeddings(torch.ones_like(image_masks))

            embeds = torch.cat([text_embeds, image_embeds], dim=1)
            masks = torch.cat([text_masks, image_masks], dim=1)
        else:
            embeds = text_embeds
            masks = text_masks

        hidden_state = embeds
        for _, blk in enumerate(self.vit.blocks):
            hidden_state, attentions = blk(hidden_state, mask=masks)

        hidden_state = self.vit.norm(hidden_state)
        text_hidden_state, image_hidden_state = (
            hidden_state[:, : text_embeds.shape[1]],
            hidden_state[:, text_embeds.shape[1] :],
        )

        pooler_output = self.pooler(hidden_state)

        return ViLTModelOutput(
            last_hidden_state=hidden_state,
            masks=masks,
            pooler_output=pooler_output,
            text_hidden_state=text_hidden_state,
            image_hidden_state=image_hidden_state,
            attentions=attentions,
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, *args, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        model = cls(*args, **kwargs)
        model.load_state_dict(state_dict, strict=False)

        assert not (
            model.text_embeddings.token_type_embeddings.weight
            == model.token_type_embeddings.weight
        ).all()

        return model
