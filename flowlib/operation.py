
"""Operation layer.

ref)
https://github.com/masa-su/pixyz/blob/master/pixyz/flows/operations.py
"""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import FlowLayer, nll_normal


class Squeeze(FlowLayer):
    """Squeeze operation: (b, c, s, s) -> (b, 4c, s/2, s/2)."""

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(batch, *)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(batch, *)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        _, channels, height, width = x.size()

        x = x.permute(0, 2, 3, 1)
        x = x.view(-1, height // 2, 2, width // 2, 2, channels)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.contiguous().view(-1, height // 2, width // 2, channels * 4)
        z = x.permute(0, 3, 1, 2)

        return z, z.new_zeros(())

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(batch, *)`.
        """

        _, channels, height, width = z.size()

        z = z.permute(0, 2, 3, 1)
        z = z.view(-1, height, width, channels // 4, 2, 2)
        z = z.permute(0, 1, 4, 2, 5, 3)
        z = z.contiguous().view(-1, 2 * height, 2 * width, channels // 4)
        x = z.permute(0, 3, 1, 2)

        return x


class Unsqueeze(Squeeze):
    """Unsqueeze operation: (b, 4c, s/2, s/2) -> (b, c, s, s)."""

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(batch, *)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(batch, *)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        return super().inverse(x), x.new_zeros(())

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(batch, *)`.
        """

        x, _ = super().forward(z)

        return x


class ChannelwiseSplit(FlowLayer):
    """Channel-wise split layer.

    Splits z -> z', h' for channel-wise.

    ref) https://github.com/openai/glow/blob/master/model.py#L545

    Args:
        in_channels (int): Number of input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels // 2, in_channels, 3, padding=1)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c//2, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        z1, z2 = torch.chunk(x, 2, dim=1)
        mu, logvar = torch.chunk(self.conv(z1), 2, dim=1)

        # Loss NLL
        logdet = nll_normal(z2, mu, F.softplus(logvar))
        logdet = logdet.sum()

        return z1, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c*2, h, w)`.
        """

        mu, logvar = torch.chunk(self.conv(z), 2, dim=1)
        z2 = mu + F.softplus(0.5 * logvar) * torch.randn_like(logvar)
        z = torch.cat([z, z2], dim=1)

        return z


class Preprocess(FlowLayer):
    """Preprocess for input images."""

    def __init__(self):
        super().__init__()

        self.register_buffer("constraint", torch.tensor([0.05]))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(batch, *)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(batch, *)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        # 1. Transform data range: [0, 1] -> [0, 255]
        x = x * 255

        # 2-1. Add noise to pixels to dequantize: [0, 255] -> [0, 1]
        x = (x + torch.randn_like(x)) / 256

        # 2-2. Transform pixel valueswith logit:  [0, 1] -> (0, 1)
        x = (1 + (2 * x - 1) * (1 - self.constraint)) / 2

        # 2-3. Logit transform
        z = x.log() - (1 - x).log()

        # Log determinant
        logdet = (
            F.softplus(z) + F.softplus(-z)
            - F.softplus(self.constraint.log() - (1 - self.constraint).log()))
        logdet = logdet.sum()

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(batch, *)`.
        """

        # Transform data range: (-inf, inf) -> (0, 1)
        x = torch.sigmoid(z)

        return x
