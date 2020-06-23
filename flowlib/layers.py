
"""Layers for flow models."""

import torch
from torch import Tensor, nn


class ActNorm2d(nn.Module):
    """Activation normalization layer.

    Args:
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.initialized = False

    def forward(self, x: Tensor) -> Tensor:
        """Forward method.

        Args:
            x (torch.Tensor):
        """
