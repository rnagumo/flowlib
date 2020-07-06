
"""Base classes for Flow models."""

from typing import Dict, Tuple, Optional

import math

import torch
from torch import Tensor, nn


class LinearZeros(nn.Linear):
    """Zero-initalized linear layer.

    Args:
        in_channels (int): Input channel size.
        out_channels (int): Output channel size.
        log_scale (float, optional): Log scale for output.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 log_scale: float = 3.0):
        super().__init__(in_channels, out_channels)

        self.log_scale = log_scale
        self.logs = nn.Parameter(torch.zeros(out_channels))

        # Initialize
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Forward calculation.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c)`.

        Returns:
            x (torch.Tensor): Output tensor, size `(b, d)`.
        """

        return super().forward(x) * (self.logs * self.log_scale).exp()


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
        z_size (tuple, optional): Tuple of latent data size, `(c, h, w)`.
        temperature (float, optional): Temperature for prior.
        conditional (bool, optional): Boolean flag for y-conditional (default =
            `False`)
        y_classes (int, optional): Number of classes in dataset.

    Attributes:
        flow_list (nn.ModuleList): Module list of `FlowLayer` classes.
    """

    def __init__(self, z_size: tuple = (3, 32, 32), temperature: float = 1.0,
                 conditional: bool = False, y_classes: int = 10):
        super().__init__()

        self.z_size = z_size
        z_channels = z_size[0]

        # Buffer for device information
        self.register_buffer("buffer", torch.zeros(1, *z_size))

        # List of flow layers, which should be overriden
        self.flow_list = nn.ModuleList()

        # Temperature for prior: (p(x))^{T^2}
        self.temperature = temperature

        # Y-conditional prior
        self.conditional = conditional
        self.prior_y = LinearZeros(y_classes, z_channels * 2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation z = f(x) with log-determinant Jacobian.

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

    def prior(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Samples prior mu and var.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            mu (torch.Tensor): Mean vector.
            var (torch.Tensor): Variance vector.
        """

        batch = x.size(0)

        if not self.conditional:
            mu = x.new_zeros((batch, *self.z_size))
            var = x.new_ones((batch, *self.z_size))
            return mu, var

        return x.new_zeros((batch, *self.z_size)), x.new_zeros((batch, *self.z_size))

    def loss_func(self, x: Tensor, y: Optional[Tensor] = None
                  ) -> Dict[str, Tensor]:
        """Loss function: -log p(x) = -log p(z) - sum log|det(dh_i/dh_{i-1})|.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            y (torch.Tensor, optional): Target label, size `(b, t)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Calculated loss.

        Raises:
            ValueError: If `self.conditional` is `True` and `y` is `None`.
        """

        if self.conditional and y is None:
            raise ValueError("y cannot be None for conditional model")

        # Inference z = f(x)
        z, logdet = self.forward(x)

        # Logdet is negative
        logdet = -logdet

        # NLL
        mu, var = self.prior(x)
        log_prob = nll_normal(z, mu, var, reduce=False)
        log_prob = log_prob.sum(dim=[1, 2, 3])

        pixels = torch.tensor(x.size()[1:]).prod().item()
        loss = ((log_prob + logdet).mean() / pixels + math.log(256)
                ) / math.log(2)

        return {"loss": loss, "log_prob": log_prob.mean(),
                "logdet": logdet.mean()}

    def sample(self, batch: int, y: Optional[Tensor] = None) -> Tensor:
        """Samples from prior.

        Args:
            batch (int): Sampled batch size.
            y (torch.Tensor, optional): Target label, size `(b, t)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        mu, var = self.prior(self.buffer.repeat(batch, 1, 1, 1))
        z = mu + self.temperature * var ** 0.5 * torch.randn_like(var)

        return self.inverse(z)

    def reconstruct(self, x: Tensor) -> Tensor:
        """Reconstructs given image: x' = f^{-1}(f(x)).

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.

        Returns:
            recon (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        z, _ = self.forward(x)
        recon = self.inverse(z)

        return recon


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
