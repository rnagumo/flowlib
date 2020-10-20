import torch
import flowlib


def test_forward() -> None:
    model = flowlib.ActNorm2d(in_channels=3)
    x = torch.randn(4, 3, 8, 8)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert not torch.isnan(z).any()
    assert logdet.size() == (1,)


def test_inverse() -> None:
    model = flowlib.ActNorm2d(in_channels=3)
    z = torch.randn(4, 3, 8, 8)
    x = model.inverse(z)

    assert x.size() == z.size()
    assert not torch.isnan(x).any()
