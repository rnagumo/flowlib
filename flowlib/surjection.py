
"""Surjection layers.

ref)
D. Nielsen+, 2020. "SurVAE Flows: Surjections to Bridge the Gap between VAEs
and Flows." Table 6.

Be carefull that `forward` method is Inverse method in Table 6, and `inverse`
method is Forward method in Table 6.
"""

from typing import Tuple

import torch
from torch import Tensor

from .base import FlowLayer, nll_normal


class Slicing(FlowLayer):
    """Slice operation with stochastic sampling.

    Args:
        sigma (float, optional): Sigma of Normal distribution.
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()

        self.register_buffer("sigma", torch.ones((1,)) * sigma)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        # z2 ~ N(x, I)
        z2 = x + torch.randn_like(x) * self.sigma

        # z = [z1, z2], z1 = x1
        z = torch.cat([x, z2], dim=1)

        logdet = nll_normal(z2, x, self.sigma ** 2, reduce=False)
        logdet = logdet.sum(dim=[1, 2, 3])

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        x, _ = torch.chunk(z, 2, dim=1)

        return x
