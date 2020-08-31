
import torch
import flowlib


def test_conv2d_forward() -> None:
    model = flowlib.Conv2dZeros(3, 3, 3, padding=1)
    x = torch.randn(4, 3, 8, 8)
    z = model(x)

    assert z.size() == x.size()
    assert not torch.isnan(z).any()


def test_convblock_forward() -> None:
    model = flowlib.ConvBlock(3, 4)
    x = torch.randn(4, 3, 8, 8)
    log_s, t = model(x)

    assert log_s.size() == x.size()
    assert not torch.isnan(log_s).any()

    assert t.size() == x.size()
    assert not torch.isnan(t).any()
