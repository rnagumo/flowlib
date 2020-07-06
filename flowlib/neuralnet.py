
"""Neural Networks.

ref)
https://github.com/masa-su/pixyz/blob/master/pixyz/layers/resnet.py
https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
"""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .normalization import ActNorm2d


class Conv2dZeros(nn.Module):
    """Weight-bias-zero-initialzed 2D convolution.

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
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.

        Returns:
            x (torch.Tensor): Convolutioned tensor, size `(b, c, h, w)`.
        """

        return self.conv(x) * (self.logs * self.log_scale).exp()


class ConvBlock(nn.Module):
    """Convolutional basic block that returns `(log_s, t)`.

    * 3 conv layers.
    * Activation normalization after convolution layer.
    * Weights of all convolution layers are initialized with zeros.
    * Bias of the last layer is initialized with zeros.

    Args:
        in_channels (int): Number of channels in input image.
        hidden_channels (int): Number of channels in mid image.
        weight_std (float, optional): Std value for weight initialization.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 weight_std: float = 0.05):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False)
        self.conv3 = Conv2dZeros(
            hidden_channels, in_channels * 2, 3, padding=1)

        self.actnorm1 = ActNorm2d(hidden_channels)
        self.actnorm2 = ActNorm2d(hidden_channels)

        # Initialize 1st and 2nd conv layer weight as normal
        self.conv1.weight.data.normal_(mean=0.0, std=weight_std)
        self.conv2.weight.data.normal_(mean=0.0, std=weight_std)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.

        Returns:
            log_s (torch.Tensor): Convoluted log_s, size `(b, c, h, w)`.
            t (torch.Tensor): Convoluted t, size `(b, c, h, w)`.
        """

        # NN
        x = self.conv1(x)
        x, _ = self.actnorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x, _ = self.actnorm2(x)
        x = F.relu(x)

        x = self.conv3(x)

        # Split output
        log_s = x[:, ::2]
        t = x[:, 1::2]

        return log_s, t


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
