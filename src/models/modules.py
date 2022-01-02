#!/usr/bin/env python

from typing import Optional

import torch
import torch.nn as nn


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
    def __init__(self, input_dim: int, activation=nn.Sigmoid()) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.gate = nn.Linear(input_dim * 2, 1)
        self.activation = activation

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.size() != b.size():
            raise ValueError("The input must have the same size.")
        if a.size(-1) != self.input_dim:
            raise ValueError("Input size must match `input_dim`.")

        gate_value = self.activation(self.gate(torch.cat([a, b], -1)))
        return gate_value * a + (1 - gate_value) * b
