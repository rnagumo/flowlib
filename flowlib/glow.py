
"""Glow model.

D. P. Kingma et al., "Glow: Generative Flow with Invertible 1×1 Convolutions"
https://arxiv.org/abs/1807.03039

ref)
https://github.com/masa-su/pixyz/blob/master/examples/glow.ipynb
"""

from typing import Tuple

import torch
from torch import Tensor, nn

from .base import FlowModel
from .conv import InvertibleConv, InvertibleConvLU
from .coupling import AffineCoupling
from .neuralnet import ResNet
from .normalization import ActNorm2d
from .operation import Squeeze, Unsqueeze, Preprocess


class ScaleTranslateNet(nn.Module):
    """Neural network for scale and translate.

    Args:
        in_channels (int): Number of channels in input image.
        mid_channels (int): Number of channels in mid image.
    """

    def __init__(self, in_channels: int, mid_channels: int):
        super().__init__()

        self.resnet = ResNet(
            in_channels, mid_channels, in_channels * 2, num_blocks=8)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.

        Returns:
            log_s (torch.Tensor): log s value, size `(b, c, h, w)`.
            t (torch.Tensor): t value, size `(b, c, h, w)`.
        """

        s_t = self.resnet(x)
        log_s, t = torch.chunk(s_t, 2, dim=1)
        log_s = torch.tanh(log_s)

        return log_s, t


class Glow(FlowModel):
    """Glow model.

    Args:
        in_channels (int, optional): Number of channels in input image.
        mid_channels (int, optional): Number of channels in mid image.
        image_size (int, optional): Size of input image.
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 64,
                 image_size: int = 32):
        super().__init__(in_size=(in_channels, image_size, image_size))

        flow_list = [Preprocess(), Squeeze()]

        # Main blocks
        for _ in range(3):
            flow_list += [
                ActNorm2d(in_channels * 4),
                InvertibleConvLU(in_channels * 4),
                AffineCoupling(
                    ScaleTranslateNet(in_channels * 4, mid_channels * 2),
                    mask_type="channel_wise", inverse_mask=False),
            ]

        flow_list += [Unsqueeze()]

        self.flow_list = nn.ModuleList(flow_list)
