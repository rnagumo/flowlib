
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
        in_channels (int): Channel size of input data.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_channels, in_channels))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        # Matrix multiplication
        weight = self.weight.view(*self.weight.size(), 1, 1)
        z = F.conv2d(x, weight)

        # Log determinant
        *_, h, w = x.size()
        _, logdet = torch.slogdet(self.weight)
        logdet = logdet.unsqueeze(0) * (h * w)

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        weight = torch.inverse(self.weight.double()).float()
        weight = weight.view(*self.weight.size(), 1, 1)
        x = F.conv2d(z, weight)

        return x


class InvertibleConvLU(FlowLayer):
    """Invertible 1 x 1 convolutional layer with LU decomposition.

    Args:
        in_channels (int): Channel size of input data.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        weight = torch.randn(in_channels, in_channels)
        nn.init.orthogonal_(weight)

        # LU decomposition
        a_lu, pivots = weight.lu()
        p, l, u = torch.lu_unpack(a_lu, pivots)
        self.register_buffer("p_mat", p)
        self.l_mat = nn.Parameter(l)
        self.u_mat = nn.Parameter(u)

        # Mask
        self.register_buffer("l_mask", torch.tril(torch.ones_like(weight), -1))
        self.register_buffer("u_mask", self.l_mask.t().clone())

        # Sign
        s = torch.diag(u)
        self.register_buffer("s_sign", torch.sign(s))
        self.s_log = nn.Parameter(s.abs().log())

        self.register_buffer("i_mat", torch.eye(in_channels))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        l = self.l_mat * self.l_mask + self.i_mat
        u = self.u_mat * self.u_mask + (self.s_sign * self.s_log.exp()).diag()
        w = self.p_mat @ l @ u
        w = w.contiguous().view(*w.size(), 1, 1)
        z = F.conv2d(x, w)

        # Log determinant
        *_, h, w = x.size()
        logdet = self.s_log.sum().unsqueeze(0) * h * w

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        l = self.l_mat * self.l_mask + self.i_mat
        u = self.u_mat * self.u_mask + (self.s_sign * self.s_log.exp()).diag()

        l_inv = l.double().inverse().float()
        u_inv = u.double().inverse().float()
        p_inv = self.p_mat.double().inverse().float()
        w = u_inv @ l_inv @ p_inv
        w = w.contiguous().view(*w.size(), 1, 1)
        x = F.conv2d(z, w)

        return x
