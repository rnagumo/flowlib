
"""Glow model.

D. P. Kingma et al., "Glow: Generative Flow with Invertible 1×1 Convolutions"
https://arxiv.org/abs/1807.03039

ref)
https://github.com/masa-su/pixyz/blob/master/examples/glow.ipynb
"""

from typing import List

from torch import nn

from .base import FlowLayer, FlowModel
from .conv import InvertibleConv
from .coupling import AffineCoupling
from .neuralnet import ConvBlock
from .normalization import ActNorm2d
from .operation import Squeeze, ChannelwiseSplit, Preprocess


class Glow(FlowModel):
    """Glow model.

    Args:
        in_channels (int, optional): Number of channels in input image.
        hidden_channels (int, optional): Number of channels in mid image.
        image_size (int, optional): Size of input image.
        depth (int, optional): Depth of flow `K`.
        level (int, optional): Number of levels `L`.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 512,
                 image_size: int = 32, depth: int = 32, level: int = 3,
                 **kwargs):
        z_size = (in_channels * 2 ** (level + 1),
                  image_size // 2 ** level, image_size // 2 ** level)
        super().__init__(z_size=z_size, **kwargs)

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
                        ConvBlock(current_channels // 2, hidden_channels)),
                ]

            # 3. Split
            if i < level - 1:
                flow_list += [ChannelwiseSplit(current_channels)]

            # Channel size is halved
            current_channels //= 2

        self.flow_list = nn.ModuleList(flow_list)
