
"""Glow model.

D. P. Kingma et al., "Glow: Generative Flow with Invertible 1×1 Convolutions"
https://arxiv.org/abs/1807.03039

ref)
https://github.com/masa-su/pixyz/blob/master/examples/glow.ipynb
"""

from typing import Tuple, List

import torch
from torch import Tensor, nn

from .base import FlowLayer, FlowModel
from .conv import InvertibleConv
from .coupling import AffineCoupling
from .neuralnet import Conv2dZeros
from .normalization import ActNorm2d
from .operation import Squeeze, ChannelwiseSplit, Preprocess


class ScaleTranslateNet(nn.Module):
    """Neural network for scale and translate.

    Args:
        in_channels (int): Number of channels in input image.
        mid_channels (int): Number of channels in mid image.
    """

    def __init__(self, in_channels: int, mid_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.ReLU(),
            Conv2dZeros(mid_channels, in_channels * 2, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation.

        Args:
            x (torch.Tensor): Input tensor, size `(b, c, h, w)`.

        Returns:
            log_s (torch.Tensor): log s value, size `(b, c, h, w)`.
            t (torch.Tensor): t value, size `(b, c, h, w)`.
        """

        s_t = self.conv(x)
        log_s, t = torch.chunk(s_t, 2, dim=1)
        log_s = torch.tanh(log_s)

        return log_s, t


class Glow(FlowModel):
    """Glow model.

    Args:
        in_channels (int, optional): Number of channels in input image.
        mid_channels (int, optional): Number of channels in mid image.
        image_size (int, optional): Size of input image.
        depth (int, optional): Depth of flow `K`.
        level (int, optional): Number of levels `L`.
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 512,
                 image_size: int = 32, depth: int = 32, level: int = 3):
        super().__init__((in_channels * 2 ** (level + 1),
                          image_size // 2 ** level, image_size // 2 ** level))

        # Current channel at each level
        current_channels = in_channels

        # Input layer
        flow_list: List[FlowLayer] = [Preprocess()]

        # Main blocks
        for i in range(level):
            current_channels *= 4

            # 1. Squeeze
            flow_list += [Squeeze()]

            # 2. K steps of flow
            for _ in range(depth):
                flow_list += [
                    ActNorm2d(current_channels),
                    InvertibleConv(current_channels),
                    AffineCoupling(
                        ScaleTranslateNet(current_channels, mid_channels),
                        mask_type="channel_wise", inverse_mask=False),
                ]

            # 3. Split
            if i < level - 1:
                flow_list += [ChannelwiseSplit(current_channels)]

            # Channel size is halved
            current_channels //= 2

        self.flow_list = nn.ModuleList(flow_list)
