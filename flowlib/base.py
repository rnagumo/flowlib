
"""Base classes for Flow models."""

from typing import Dict, Tuple

import math

import torch
from torch import Tensor, nn


class FlowLayer(nn.Module):
    """Base class for Flow layers."""

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

        Args:
            x (torch.Tensor): Observations, size `(batch, *)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(batch, *)`.
            logdet (torch.Tensor): Log determinant Jacobian, size `(batch, *)`.
        """

        raise NotImplementedError

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(batch, *)`.
        """

        raise NotImplementedError


class FlowModel(nn.Module):
    """Base class for Flow models.

    Attributes:
        flow_list (nn.ModuleList): Module list of `FlowLayer` classes.
    """

    def __init__(self, z_size: tuple = (1,)):
        super().__init__()

        # List of flow layers, which should be overriden
        self.flow_list = nn.ModuleList()

        # Prior p(z)
        self._prior_mu = torch.zeros(z_size)
        self._prior_var = torch.ones(z_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation z = f(x).

        Args:
            x (torch.Tensor): Observations, size `(batch, *)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(batch, *)`.
        """

        z, _ = self.inference(x)

        return z

    def loss_func(self, x: Tensor) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            x (torch.Tensor): Observations, size `(batch, *)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Calculated loss.
        """

        z, logdet = self.inference(x)
        log_prob = nll_normal(z, self._prior_mu, self._prior_var)
        loss = (log_prob + logdet).mean()

        return {"loss": loss}

    def inference(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Inferences latents and calculates loss.

        Args:
            x (torch.Tensor): Observations, size `(batch, *)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(batch, *)`.
            logdet (torch.Tensor): Log determinant Jacobian, size `(batch, *)`.
        """

        logdet = x.new_zeros((x.size(0),))

        for flow in self.flow_list:
            x, _logdet = flow(x)
            logdet += _logdet

        return x, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(batch, *)`.
        """

        for flow in self.flow_list[::-1]:
            z = flow.inverse(z)

        return z

    def sample(self, batch: int) -> Tensor:
        """Samples from prior.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(batch, *)`.
        """

        var = torch.cat([self._prior_var.unsqueeze(0)] * batch)
        z = self._prior_mu + var ** 0.5 * torch.randn_like(var)

        return self.inverse(z)


def nll_normal(x: Tensor, mu: Tensor, var: Tensor, reduce: bool = True
               ) -> Tensor:
    """Negative log likelihood for 1-D Normal distribution.

    Args:
        x (torch.Tensor): Inputs tensor, size `(*, dim)`.
        mu (torch.Tensor): Mean vector, size `(*, dim)`.
        var (torch.Tensor): Variance vector, size `(*, dim)`.
        reduce (bool, optional): If `True`, sum calculated loss for each
            data point.

    Returns:
        nll (torch.Tensor): Calculated nll for each data, size `(*,)` if
            `reduce` is `True`, `(*, dim)` otherwise.
    """

    nll = 0.5 * ((2 * math.pi * var).log() + (x - mu) ** 2 / var)

    if reduce:
        return nll.sum(-1)
    return nll
