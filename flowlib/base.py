
"""Base classes for Flow models."""

from typing import Dict, Tuple, Optional

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
        y_classes (int, optional): Number of classes in dataset.
        y_weight (float, optional): Weight for classification loss.

    Attributes:
        flow_list (nn.ModuleList): Module list of `FlowLayer` classes.
    """

    def __init__(self, z_size: tuple = (3, 32, 32), temperature: float = 1.0,
                 y_classes: int = 10, y_weight: float = 0.01):
        super().__init__()

        # List of flow layers, which should be overriden
        self.flow_list = nn.ModuleList()

        # Latent size
        self.z_size = z_size

        # Buffer for prior parameters
        self.register_buffer("buffer", torch.zeros(1, *z_size))

        # Temperature for prior: (p(x))^{T^2}
        self.temperature = temperature

        # Y-conditional prior
        self.y_classes = y_classes
        self.y_weight = y_weight

        z_channels = z_size[0]
        self.y_prior = LinearZeros(y_classes, z_channels * 2)
        self.y_projector = LinearZeros(z_channels, y_classes)

        # Criterion for classification loss
        self.criterion = nn.BCEWithLogitsLoss()

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

    def prior(self, batch: int, y: Optional[Tensor] = None
              ) -> Tuple[Tensor, Tensor]:
        """Samples prior mu and var.

        Args:
            batch (int): Batch size of prior.
            y (torch.Tensor, optional): Target label, size `(b,)`.

        Returns:
            mu (torch.Tensor): Mean vector, size `(b, c, 1, 1)`.
            var (torch.Tensor): Variance vector, size `(b, c, 1, 1)`.

        Raises:
            ValueError: If batch size does not equal size of y.
        """

        if y is None:
            mu = self.buffer.new_zeros((batch, *self.z_size))
            var = self.buffer.new_ones((batch, *self.z_size))
            return mu, var

        if batch != y.size(0):
            raise ValueError(f"Incompatible size: y batch size {y.size(0)} "
                             f"should be the same as batch size {batch}")

        # One-hot encoding
        y = F.one_hot(y, num_classes=self.y_classes)

        # Inference
        h = self.y_prior(y.float())
        mu, logvar = torch.chunk(h, 2, dim=-1)
        var = F.softplus(logvar)

        # Fix size: (b, c) -> (b, c, 1, 1)
        mu = mu.contiguous().view(batch, -1, 1, 1)
        var = var.contiguous().view(batch, -1, 1, 1)

        # Expand height and width: (b, c, 1, 1) -> (b, c, h, w)
        *_, h, w = self.z_size
        mu = mu.repeat(1, 1, h, w)
        var = var.repeat(1, 1, h, w)

        return mu, var

    def loss_func(self, x: Tensor, y: Optional[Tensor] = None
                  ) -> Dict[str, Tensor]:
        """Loss function.

        loss = -log p(z) - log|det(dh_i/dh_{i-1})| + lmd * cross_entropy.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            y (torch.Tensor, optional): Target label, size `(b,)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Calculated loss.
        """

        # Inference z = f(x)
        z, logdet = self.forward(x)

        # Negative logdet
        logdet = -logdet

        # NLL of prior
        mu, var = self.prior(x.size(0), y)
        log_prob = nll_normal(z, mu, var, reduce=False)
        log_prob = log_prob.sum(dim=[1, 2, 3])

        # Loss in bits per dimension
        pixels = torch.tensor(x.size()[1:]).prod().item()
        nll = ((log_prob + logdet) / pixels + math.log(256)) / math.log(2)

        # Classification loss
        if y is None:
            loss_classes = x.new_zeros((1,))
        else:
            y_logits = self.y_projector(z.mean(dim=[2, 3]))
            y = F.one_hot(y, num_classes=self.y_classes).float()
            loss_classes = self.y_weight * self.criterion(y_logits, y)

        # Returned loss
        loss = nll + loss_classes

        return {"loss": loss.mean(), "log_prob": log_prob.mean(),
                "logdet": logdet.mean(), "classification": loss_classes.mean()}

    def sample(self, batch: int, y: Optional[Tensor] = None) -> Tensor:
        """Samples from prior.

        Args:
            batch (int): Sampled batch size.
            y (torch.Tensor, optional): Target label, size `(b,)`.

        Returns:
            x (torch.Tensor): Decoded Observations, size `(b, c, h, w)`.
        """

        mu, var = self.prior(batch, y)
        z = mu + self.temperature * (var ** 0.5) * torch.randn_like(var)

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
