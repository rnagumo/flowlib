
"""Neural Networks.

ref)
https://github.com/masa-su/pixyz/blob/master/pixyz/layers/resnet.py
"""

import torch
from torch import Tensor, nn


class Conv2dZeros(nn.Module):
    """Weight-zero-initialzed 2D convolution.

    Args:
        in_channels (int): Input channel size.
        out_channels (int): Output channel size.
        kernel_size (int): Kernel size.
        padding (int): Padding size.
        bias (bool, optional): Boolean flag for bias in conv.
        log_scale (float, optional): Log scale for output.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int, bias: bool = True, log_scale: float = 3.0):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.log_scale = log_scale
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward convolution.

        Args:
            x (torch.Tensor): Input tensor, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Convolutioned tensor, size `(batch, *)`.
        """

        return self.conv(x) * (self.logs * self.log_scale).exp()


class ResidualBlock(nn.Module):
    """ResNet basic block.

    The last convolution is initialized with zeros.

    Args:
        in_channels (int): Number of channels in input image.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            Conv2dZeros(in_channels, in_channels, 3, padding=1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x (torch.Tensor): Input tensor, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Convolutioned tensor, size `(batch, *)`.
        """

        return x + self.conv(x)
