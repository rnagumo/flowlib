
from typing import Tuple

import torch
from torch import Tensor

import flowlib


class TempNet(torch.nn.Module):
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return x, x


def affine_coupling_forward() -> None:
    scale_trans_net = TempNet()
    model = flowlib.AffineCoupling(scale_trans_net)
    x = torch.randn(4, 4, 8, 8)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert not torch.isnan(z).any()
    assert logdet.size() == (4,)


def affine_coupling_inverse() -> None:
    scale_trans_net = TempNet()
    model = flowlib.AffineCoupling(scale_trans_net)
    z = torch.randn(4, 4, 8, 8)
    x = model.inverse(z)

    assert x.size() == z.size()
    assert not torch.isnan(x).any()


def _base_case(mask_type, inverse_mask) -> None:
    scale_trans_net = TempNet()
    model = flowlib.MaskedAffineCoupling(
        scale_trans_net, mask_type=mask_type, inverse_mask=inverse_mask)

    # Forward
    x = torch.randn(4, 3, 8, 8)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert not torch.isnan(z).any()
    assert logdet.size() == (4,)

    # Inverse
    z = torch.randn(4, 3, 8, 8)
    x = model.inverse(z)

    assert x.size() == z.size()
    assert not torch.isnan(x).any()


def test_checkerboard_true() -> None:
    _base_case("checkerboard", True)


def test_checkerboard_false() -> None:
    _base_case("checkerboard", False)


def test_channel_wise_true() -> None:
    _base_case("channel_wise", True)


def test_channel_wise_false() -> None:
    _base_case("channel_wise", False)


def test_checkerboard_mask() -> None:

    mask = flowlib.checkerboard_mask(3, 4, inverse=False)
    true_mask = torch.tensor([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [1, 0, 1, 0]])
    assert (mask == true_mask).all()

    mask = flowlib.checkerboard_mask(3, 4, inverse=True)
    true_mask = torch.tensor([[0, 1, 0, 1],
                              [1, 0, 1, 0],
                              [0, 1, 0, 1]])
    assert (mask == true_mask).all()


def test_channel_wise_mask() -> None:

    mask = flowlib.channel_wise_mask(6, inverse=False)
    assert (mask[:3] == 1).all()
    assert (mask[3:] == 0).all()

    mask = flowlib.channel_wise_mask(6, inverse=True)
    assert (mask[:3] == 0).all()
    assert (mask[3:] == 1).all()
