
"""Coupling layer.

ref)
https://github.com/masa-su/pixyz/blob/master/pixyz/flows/coupling.py
"""

from typing import Tuple

import torch
from torch import Tensor, nn

from .base import FlowLayer


class AffineCoupling(FlowLayer):
    """Affine coupling layer.

    Args:
        scale_trans_net (nn.Module): Function for scale and transition.
        mask_type (str, optional): Mask type (checkerboard or channel_wise).
        inverse_mask (bool, optional): If `True`, reverse mask.
    """

    def __init__(self, scale_trans_net: nn.Module,
                 mask_type: str = "channel_wise", inverse_mask: bool = False):
        super().__init__()

        if mask_type not in ["checkerboard", "channel_wise"]:
            raise ValueError(
                "Mask type should be 'checkerboard' or 'channel_wise'")

        self.scale_trans_net = scale_trans_net
        self.mask_type = mask_type
        self.inverse_mask = inverse_mask

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(batch, *)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(batch, *)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        mask = self._generate_mask(x)
        x_a = x * (1 - mask)
        x_b = x * mask

        log_s, t = self.scale_trans_net(x_b)
        log_s = log_s * (1 - mask)
        t = t * (1 - mask)
        z = (x_a * log_s.exp() + t) + x_b

        # Log determinant
        logdet = log_s.sum()

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(batch, *)`.
        """

        mask = self._generate_mask(z)
        z_a = z * (1 - mask)
        z_b = z * mask

        log_s, t = self.scale_trans_net(z_b)
        log_s = log_s * (1 - mask)
        t = t * (1 - mask)
        x = (z_a - t) * (-log_s).exp() + z_b

        return x

    def _generate_mask(self, x: Tensor) -> Tensor:
        """Generates mask given inputs.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            mask (torch.Tensor): Generated mask.
        """

        if x.dim() == 4:
            _, channel, height, width = x.size()
            if self.mask_type == "checkerboard":
                mask = checkerboard_mask(height, width, self.inverse_mask)
                mask = mask.view(1, 1, height, width)
            else:
                mask = channel_wise_mask(channel, self.inverse_mask)
                mask = mask.view(1, channel, 1, 1)
        elif x.dim() == 2 and self.mask_type == "channel_wise":
            _, n_features = x.size()
            mask = channel_wise_mask(n_features, self.inverse_mask)
            mask = mask.view(1, n_features)
        else:
            raise ValueError("Invalid dimension and mask type")

        return mask.to(x.device)


def checkerboard_mask(height: int, width: int, inverse: bool = False
                      ) -> Tensor:
    """Checker board mask pattern.

    Example (inverse=False):

    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 0, 1, 0]]

    Args:
        height (int): Height of 2D.
        width (int): Width of 2D.
        inverse (bool, optional): If `True`, mask pattern is reversed.

    Returns:
        mask (torch.Tensor): Generated mask pattern, size `(height, width)`.
    """

    mask = (torch.arange(height).view(-1, 1) + torch.arange(width)) % 2
    if not inverse:
        mask = 1 - mask

    return mask


def channel_wise_mask(channel: int, inverse: bool = False) -> Tensor:
    """Channel wise mask pattern.

    Args:
        channel (int): Number of channels.

    inverse (bool, optional): If `True`, mask pattern is reversed.

    Returns:
        mask (torch.Tensor): Generated mask pattern, size `(channel,)`.
    """

    mask = torch.zeros(channel)
    if inverse:
        mask[channel // 2:] = 1
    else:
        mask[:channel // 2] = 1

    return mask
