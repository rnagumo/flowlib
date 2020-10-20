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

    ref) https://github.com/openai/glow/blob/master/model.py

    Difference between original paper and official code.

    * Squash function of log(s): exp -> sigmoid
    * Affine coupling: s * x + t -> (x + t) * s

    Args:
        scale_trans_net (nn.Module): Function for scale and transition.
    """

    def __init__(self, scale_trans_net: nn.Module) -> None:
        super().__init__()

        self.scale_trans_net = scale_trans_net

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        channels = x.size(1)
        if channels % 2 != 0:
            raise ValueError("Channel number should be even.")

        x_a = x[:, : channels // 2]
        x_b = x[:, channels // 2 :]

        log_s, t = self.scale_trans_net(x_b)
        scale = torch.sigmoid(log_s + 2)
        z = torch.cat([((x_a + t) * scale), x_b], dim=1)

        logdet = scale.log().sum(dim=[1, 2, 3])

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:

        channels = z.size(1)
        if channels % 2 != 0:
            raise ValueError("Channel number should be even.")

        z_a = z[:, : channels // 2]
        z_b = z[:, channels // 2 :]

        log_s, t = self.scale_trans_net(z_b)
        scale = torch.sigmoid(log_s + 2)
        x = torch.cat([z_a / scale - t, z_b], dim=1)

        return x


class MaskedAffineCoupling(FlowLayer):
    """Masked affine coupling layer.

    Args:
        scale_trans_net (nn.Module): Function for scale and transition.
        mask_type (str, optional): Mask type (checkerboard or channel_wise).
        inverse_mask (bool, optional): If `True`, reverse mask.
    """

    def __init__(
        self,
        scale_trans_net: nn.Module,
        mask_type: str = "channel_wise",
        inverse_mask: bool = False,
    ) -> None:
        super().__init__()

        if mask_type not in ["checkerboard", "channel_wise"]:
            raise ValueError("Mask type should be 'checkerboard' or 'channel_wise'")

        self.scale_trans_net = scale_trans_net
        self.mask_type = mask_type
        self.inverse_mask = inverse_mask

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        mask = self._generate_mask(x)
        x_a = x * (1 - mask)
        x_b = x * mask

        log_s, t = self.scale_trans_net(x_b)
        log_s = torch.sigmoid(log_s + 2) * (1 - mask)
        t = t * (1 - mask)
        z = (x_a * log_s.exp() + t) + x_b

        # Log determinant
        logdet = log_s.sum(dim=[1, 2, 3])

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:

        mask = self._generate_mask(z)
        z_a = z * (1 - mask)
        z_b = z * mask

        log_s, t = self.scale_trans_net(z_b)
        log_s = torch.sigmoid(log_s + 2) * (1 - mask)
        t = t * (1 - mask)
        x = (z_a - t) * (-log_s).exp() + z_b

        return x

    def _generate_mask(self, x: Tensor) -> Tensor:

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


def checkerboard_mask(height: int, width: int, inverse: bool = False) -> Tensor:
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
        mask[channel // 2 :] = 1
    else:
        mask[: channel // 2] = 1

    return mask
