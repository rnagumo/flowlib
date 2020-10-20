import torch
import flowlib


def test_invertible_conv_forward() -> None:
    model = flowlib.InvertibleConv(in_channels=3)
    x = torch.randn(4, 3, 8, 8)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert not torch.isnan(z).any()
    assert logdet.size(), (1,)


def test_invertible_conv_inverse() -> None:
    model = flowlib.InvertibleConv(in_channels=3)
    z = torch.randn(4, 3, 8, 8)
    x = model.inverse(z)

    assert x.size() == z.size()
    assert not torch.isnan(x).any()


def test_invertible_convlu_forward() -> None:
    model = flowlib.InvertibleConvLU(in_channels=3)
    x = torch.randn(4, 3, 8, 8)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert not torch.isnan(z).any()
    assert logdet.size() == (1,)


def test_invertible_convlu_inverse() -> None:
    model = flowlib.InvertibleConvLU(in_channels=3)
    z = torch.randn(4, 3, 8, 8)
    x = model.inverse(z)

    assert x.size() == z.size()
    assert not torch.isnan(x).any()
