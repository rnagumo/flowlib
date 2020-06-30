
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
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        raise NotImplementedError

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        raise NotImplementedError


class FlowModel(nn.Module):
    """Base class for Flow models.

    Args:
        in_size (tuple, optional): Tuple of input data size.

    Attributes:
        flow_list (nn.ModuleList): Module list of `FlowLayer` classes.
    """

    def __init__(self, in_size: tuple = (1,)):
        super().__init__()

        # List of flow layers, which should be overriden
        self.flow_list = nn.ModuleList()

        # Prior p(z)
        self.register_buffer("_prior_mu", torch.zeros(in_size))
        self.register_buffer("_prior_var", torch.ones(in_size))

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation z = f(x).

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c, h, w)`.
        """

        z, _ = self.inference(x)

        return z

    def loss_func(self, x: Tensor) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Calculated loss.
        """

        z, logdet = self.inference(x)

        # Loss is -log p(x), so logdet should be negative
        logdet = -logdet

        # NLL
        log_prob = nll_normal(z, self._prior_mu, self._prior_var, reduce=False)
        log_prob = log_prob.sum(dim=[1, 2, 3])

        pixels = torch.tensor(x.size()[1:]).prod().item()
        loss = ((logdet + logdet).mean() / pixels + math.log(256)
                ) / math.log(2)

        return {"loss": loss, "log_prob": log_prob.mean(),
                "logdet": logdet.mean()}

    def inference(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Inferences latents and calculates loss.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            z (torch.Tensor): Encoded latents, size `(b, c, h, w)`.
            logdet (torch.Tensor): Log determinant Jacobian.
        """

        logdet = x.new_zeros((x.size(0),))

        for flow in self.flow_list:
            x, _logdet = flow(x)
            logdet += _logdet

        return x, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse propagation x = f^{-1}(z).

        Args:
            z (torch.Tensor): latents, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        for flow in self.flow_list[::-1]:
            z = flow.inverse(z)

        return z

    def sample(self, batch: int) -> Tensor:
        """Samples from prior.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
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
