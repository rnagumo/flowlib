
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
        in_channel (int): Channel size of input data.
    """

    def __init__(self, in_channel: int):
        super().__init__()

        in_size = (1, in_channel, 1, 1)
        self.weight = nn.Parameter(torch.ones(in_size))
        self.bias = nn.Parameter(torch.zeros(in_size))
        self.initialized = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward method.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        if not self.initialized:
            self.initialize_parameters(x)

        z = x + self.bias
        z = z * torch.exp(self.weight)

        # Compute Jacobian
        *_, h, w = x.size()
        logdet = self.weight.sum() * h * w

        return x, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        if not self.initialized:
            self.initialize_parameters(z)

        x = z * torch.exp(-self.weight)
        x = x - self.bias

        return x

    def initialize_parameters(self, x: Tensor, eps: float = 1e-8) -> None:
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
            var = torch.mean(x.clone() ** 2, dim=[0, 2, 3], keepdim=True)
            weight = torch.log(1.0 / (torch.sqrt(var) + eps))
            self.weight.data.copy_(weight)
            self.bias.data.copy_(bias)
            self.initialized = True
