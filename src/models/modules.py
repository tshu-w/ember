#!/usr/bin/env python

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .utils import split_into_two_factors


class Fusion(nn.Module):
    # https://arxiv.org/pdf/1512.08422.pdf
    def __init__(
        self, input_dim: int, output_dim: Optional[int] = None, activation=nn.ReLU()
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim

        self.fusion = nn.Sequential(
            nn.Linear(self.input_dim * 4, self.output_dim),
            activation,
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if a.size() != b.size():
            raise ValueError("The input must have the same size.")
        if a.size(-1) != self.input_dim:
            raise ValueError("Input size must match `input_dim`.")

        return self.fusion(torch.cat([a, b, a - b, a * b], dim=1))


class GatedSum(nn.Module):
    def __init__(self, input_dim: int, activation=torch.nn.Sigmoid()) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.gate = torch.nn.Linear(input_dim * 2, 1)
        self.activation = activation

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.size() != b.size():
            raise ValueError("The input must have the same size.")
        if a.size(-1) != self.input_dim:
            raise ValueError("Input size must match `input_dim`.")

        gate_value = self.activation(self.gate(torch.cat([a, b], -1)))
        return gate_value * a + (1 - gate_value) * b


class ImageEncoder(nn.Module):
    def __init__(self, feature_type=Literal["grid", "roi", "e2e"], num_image_embeds=1):
        super().__init__()

        self.feature_type = feature_type

        if feature_type != "e2e":
            self.model = nn.Identity()
        else:
            model = torchvision.models.resnet152(pretrained=True)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)

        if feature_type == "roi":
            self.num_image_embeds = num_image_embeds
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveAvgPool2d(split_into_two_factors(num_image_embeds))

    def forward(self, x):
        # grid: BxDx7x7 -> BxDxN -> BxNxD
        # roi: BxNxDx7x7 -> BxNxDx1x1 -> BxNxD
        # e2e: Bx3x224x224 -> BxDx7x7 -> BxDxN -> BxNxD

        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)

        if self.feature_type == "roi":
            out = out[:, : self.num_image_embeds, :]
        else:
            out = out.transpose(1, 2).contiguous()

        return out  # BxNxD


class TextMatcher(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        type: Literal["cross", "dual"],
    ):
        super().__init__()

        self.model = model
        self.type = type

        if self.type == "dual":
            self.fusion = Fusion(input_dim=model.config.hidden_size)

        self.output_dim = model.config.hidden_size

    def forward(self, x):
        if self.type == "cross":
            return self.model(**x, return_dict=True).pooler_output

        elif self.type == "dual":
            assert len(x) == 2

            a = self.model(**x[0], return_dict=True).pooler_output
            b = self.model(**x[1], return_dict=True).pooler_output

            return self.fusion(a, b)

    def get_output_dim(self):
        return self.output_dim


class ImageMatcher(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        self.model = model
        self.fusion = Fusion(input_dim=input_dim, output_dim=output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim

    def forward(self, x):
        assert x.shape[1] == 2

        a = x[:, 0, :].squeeze(dim=1)
        a = self.model(a).flatten(start_dim=1)

        b = x[:, 1, :].squeeze(dim=1)
        b = self.model(b).flatten(start_dim=1)

        a = F.pad(a, (0, self.input_dim - a.size(1)))
        b = F.pad(b, (0, self.input_dim - b.size(1)))

        return self.fusion(a, b)

    def get_output_dim(self):
        return self.output_dim
