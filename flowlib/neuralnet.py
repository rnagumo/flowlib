
"""Neural Networks.

ref)
https://github.com/masa-su/pixyz/blob/master/pixyz/layers/resnet.py
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class WNConv2d(nn.Module):
    """Weight-normalized 2D convolution.

    Args:
        in_channels (int): Input channel size.
        out_channels (int): Output channel size.
        kernel_size (int): Kernel size.
        padding (int): Padding size.
        bias (bool, optional): Boolean flag for bias in conv.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int, bias: bool = True):
        super().__init__()

        self.conv = nn.utils.weight_norm(nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=bias
        ))

    def forward(self, x: Tensor) -> Tensor:
        """Forward convolution.

        Args:
            x (torch.Tensor): Input tensor, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Convolutioned tensor, size `(batch, *)`.
        """

        return self.conv(x)


class ResidualBlock(nn.Module):
    """ResNet basic block with weight normalization.

    Args:
        in_channels (int): Number of channels in input image.
        out_channels (int): Number of channels in output image.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            WNConv2d(in_channels, out_channels, 3, 1, False),
            nn.BatchNorm2d(out_channels),
            WNConv2d(out_channels, out_channels, 3, 1, True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x (torch.Tensor): Input tensor, size `(batch, *)`.

        Returns:
            x (torch.Tensor): Convolutioned tensor, size `(batch, *)`.
        """

        return x + self.conv(x)


class ResNet(nn.Module):
    """ResNet layer for scale and translate factors.

    Args:
        in_channels (int): Number of channels in input image.
        mid_channels (int): Number of channels in mid image.
        out_channels (int): Number of channels in output image.
        num_block (int): Number of residual blocks.
        kernel_size (int, optional): Size of convolving kernel.
        padding (int, optional): Zero-padding size.
        double_after_norm (bool, optional): Boolean flag for doubling input.
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int,
                 num_blocks: int, kernel_size: int = 3, padding: int = 1,
                 double_after_norm: bool = True):
        super().__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        self.double_after_norm = double_after_norm

        self.in_conv = WNConv2d(
            in_channels * 2, mid_channels, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(
            mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        self.blocks = nn.ModuleList([
            ResidualBlock(mid_channels, mid_channels)
            for _ in range(num_blocks)])
        self.skips = nn.ModuleList([
            WNConv2d(mid_channels, mid_channels, 1, 0, bias=True)
            for _ in range(num_blocks)])

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(
            mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation.

        Args:
            x (torch.Tensor): Input tensor, size `(batch, *)`.

        Returns:
            output (torch.Tensor): Output tensor, size `(batch, *)`.
        """

        x = self.in_norm(x)
        if self.double_after_norm:
            x = x * 2

        x = torch.cat([x, -x], dim=1)
        x = F.relu(x)
        x = self.in_conv(x)
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)

        return x
