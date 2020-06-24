
"""Convolutional layers.

ref)
https://github.com/masa-su/pixyz/blob/master/pixyz/flows/conv.py
https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
"""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import FlowLayer


class InvertibleConv(FlowLayer):
    """Invertible 1 x 1 convolutional layer.

    Args:
        in_channel (int): Channel size of input data.
    """

    def __init__(self, in_channel: int):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(in_channel, in_channel))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(batch, *)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(batch, *)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        # Matrix multiplication
        weight = self.weight.view(*self.weight.size(), 1, 1)
        z = F.conv2d(x, weight)

        # Log determinant
        *_, h, w = x.size()
        _, logdet = torch.slogdet(self.weight)
        logdet = logdet * (h * w)

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(batch, *)`.
        """

        weight = torch.inverse(self.weight.double()).float()
        weight = weight.view(*self.weight.size(), 1, 1)
        x = F.conv2d(z, weight)

        return x
