"""Normalization layers.

ref)
https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
https://github.com/masa-su/pixyz/blob/master/pixyz/flows/normalizations.py
https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py#L264
"""

from typing import Tuple

import torch
from torch import Tensor, nn

from .base import FlowLayer


class ActNorm2d(FlowLayer):
    """Activation normalization layer for 2D data.

    Args:
        in_channels (int): Channel size of input data.
        scale (float, optional): Scaling factor.
    """

    def __init__(self, in_channels: int, scale: float = 1.0) -> None:
        super().__init__()

        in_size = (1, in_channels, 1, 1)
        self.weight = nn.Parameter(torch.ones(in_size))
        self.bias = nn.Parameter(torch.zeros(in_size))
        self.scale = scale
        self.initialized = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        if not self.initialized:
            self.initialize_parameters(x)

        z = x + self.bias
        z = z * torch.exp(self.weight)

        *_, h, w = x.size()
        logdet = self.weight.sum().unsqueeze(0) * h * w

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:

        if not self.initialized:
            self.initialize_parameters(z)

        x = z * torch.exp(-self.weight)
        x = x - self.bias

        return x

    def initialize_parameters(self, x: Tensor, eps: float = 1e-6) -> None:
        """Initializes parameters with first given data.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            eps (float, optional): Small noise.
        """

        if not self.training:
            return

        with torch.no_grad():
            # Channel-wise mean and var
            bias = -torch.mean(x.clone(), dim=[0, 2, 3], keepdim=True)
            var = torch.mean((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            weight = torch.log(self.scale / (torch.sqrt(var) + eps))
            self.weight.data.copy_(weight)
            self.bias.data.copy_(bias)
            self.initialized = True
