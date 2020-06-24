
"""Layers for flow models."""

from typing import Tuple

import torch
from torch import Tensor, nn

from .base import FlowLayer


class ActNorm2d(FlowLayer):
    """Activation normalization layer for 2D data.

    Args:
        in_channel (int): Channel size of input data.
        scale (float, optional): Scale parameter.
    """

    def __init__(self, in_channel: int, scale: float = 1.0):
        super().__init__()

        in_size = (1, in_channel, 1, 1)
        self.register_parameter("bias", nn.Parameter(torch.zeros(in_size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(in_size)))
        self.scale = float(scale)
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

        x = self._center(x, inverse=False)
        x, logdet = self._scale(x, inverse=False)

        return x, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(batch, *)`.
        """

        if not self.initialized:
            self.initialize_parameters(z)

        z, _ = self._scale(z, inverse=True)
        z = self._center(z, inverse=True)

        return z

    def _center(self, x: Tensor, inverse: bool) -> Tensor:
        """Calculates mean.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            inverse (bool): If `True`, inverse propagation.

        Args:
            x (torch.Tensor): Biased tensor, size `(b, c, h, w)`.
        """

        # Forward
        if not inverse:
            return x + self.bias

        # Inverse
        return x - self.bias

    def _scale(self, x: Tensor, inverse: bool) -> Tuple[Tensor, Tensor]:
        """Calculates mean.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            inverse (bool): If `True`, inverse propagation.

        Args:
            x (torch.Tensor): Scaled tensor, size `(b, c, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        if not inverse:
            # Forward
            x = x * torch.exp(self.logs)
        else:
            # Inverse
            x = x * torch.exp(-self.logs)

        # Compute Jacobian
        *_, h, w = x.size()
        logdet = self.logs.sum() * h * w

        return x, logdet

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
            logs = torch.log(self.scale / (torch.sqrt(var) + eps))
            self.bias.data.copy_(bias)
            self.logs.data.copy_(logs)
            self.initialized = True
