
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

from .base import FlowLayer, nll_bernoulli, nll_normal


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


class AbsSurjection(FlowLayer):
    """Absolute generative surjection."""

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        s = torch.bernoulli(torch.sigmoid(x))
        z = s * x

        # Log det
        logdet = nll_bernoulli(s, torch.sigmoid(x), reduce=False)
        logdet = logdet.sum(dim=[1, 2, 3])

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        return z.abs()


class MaxSurjection(FlowLayer):
    """Max generative surjection layer."""

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        # Reshape
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)

        # Sample z_k
        dist = torch.distributions.Categorical(logits=x)
        indices = dist.sample()
        indices = indices.unsqueeze(-1)
        z_k = x.gather(-1, indices)

        # Sample z_{\k} in range of [0, z_k]
        z = torch.rand_like(x) * z_k

        # Fill values
        idx1 = torch.arange(b).view(-1, 1, 1)
        idx2 = torch.arange(c).view(1, -1, 1)
        z[idx1, idx2, indices] = z_k

        # Revert size: (b, c, h*w) -> (b, c, h, w)
        z = z.view(b, c, h, w)

        # Log det
        q_k = dist.log_prob(indices.squeeze(-1))
        q_k = q_k.sum(dim=-1)

        q_z = nll_bernoulli(z / z_k.unsqueeze(-1), torch.ones_like(z), False)
        q_z = q_z.sum(dim=[1, 2, 3])

        logdet = q_k + q_z

        return z, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        # Reshape
        b, c, h, w = z.size()
        z = z.view(b, c, h * w)

        # Indices of argmax
        indices = z.argmax(dim=-1)
        indices = indices.unsqueeze(-1)

        # Fill values
        x = torch.zeros_like(z)
        idx1 = torch.arange(b).view(-1, 1, 1)
        idx2 = torch.arange(c).view(1, -1, 1)
        x[idx1, idx2, indices] = z[idx1, idx2, indices]

        # Revert data size
        x = x.view(b, c, h, w)

        return x
