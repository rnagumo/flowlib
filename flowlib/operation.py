
"""Operation layer.

ref)
https://github.com/masa-su/pixyz/blob/master/pixyz/flows/operations.py
"""

from typing import Tuple

from torch import Tensor

from .base import FlowLayer


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
