import torch
import flowlib


def test_forward() -> None:
    model = flowlib.Glow()
    x = torch.rand(2, 3, 32, 32)
    z, logdet = model(x)

    assert z.size() == (2, 48, 4, 4)
    assert not torch.isnan(z).any()
    assert logdet.size() == (2,)


def test_inverse() -> None:
    model = flowlib.Glow()
    # Initialize actnorm by forward step
    x = torch.rand(2, 3, 32, 32)
    model(x)

    z = torch.randn(2, 48, 4, 4)
    x = model.inverse(z)

    assert x.size() == (2, 3, 32, 32)
    assert not torch.isnan(x).any()


def test_loss_func() -> None:
    model = flowlib.Glow()
    x = torch.randn(2, 3, 32, 32)
    loss_dict = model.loss_func(x)

    assert isinstance(loss_dict, dict)
    assert loss_dict["loss"] != 0
    assert loss_dict["log_prob"] != 0
    assert loss_dict["logdet"] != 0
    assert loss_dict["classification"] == 0


def test_loss_func_conditional() -> None:
    model = flowlib.Glow()
    x = torch.randn(2, 3, 32, 32)
    y = torch.arange(2)
    loss_dict = model.loss_func(x, y)

    assert isinstance(loss_dict, dict)
    assert loss_dict["loss"] != 0
    assert loss_dict["log_prob"] != 0
    assert loss_dict["logdet"] != 0
    assert loss_dict["classification"] != 0


def test_sample() -> None:
    model = flowlib.Glow()
    # Initialize actnorm by forward step
    x = torch.rand(2, 3, 32, 32)
    model(x)

    x = model.sample(5)
    assert x.size() == (5, 3, 32, 32)
    assert not torch.isnan(x).any()


def test_inference_with_other_shape() -> None:
    model = flowlib.Glow(3, 64, image_size=64, depth=12, level=2)
    x = torch.rand(2, 3, 64, 64)
    z, logdet = model(x)

    assert z.size() == (2, 24, 16, 16)
    assert not torch.isnan(z).any()
    assert logdet.size() == (2,)
