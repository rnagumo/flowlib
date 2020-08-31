
import pytest

import torch
from torch import nn

import flowlib


def test_linear_zeros_forward() -> None:
    model = flowlib.LinearZeros(3, 4)
    x = torch.randn(4, 3)
    z = model(x)

    assert z.size() == (4, 4)
    assert not torch.isnan(z).any()


# Temporal layer class
class TempLayer(nn.Module):
    def forward(self, x):
        return x, x.abs().mean().unsqueeze(0)

    def inverse(self, z):
        return z


@pytest.fixture
def model() -> nn.Module:
    _model = flowlib.FlowModel()
    _model.flow_list = nn.ModuleList([
        TempLayer(), TempLayer()
    ])
    return _model


def test_example_forward(model) -> None:
    x = torch.randn(2, 3, 32, 32)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert logdet.size() == (2,)


def test_example_prior(model) -> None:
    mu, var = model.prior(2)

    assert mu.size() == (2, 3, 32, 32)
    assert var.size() == (2, 3, 32, 32)

    # Conditional
    y = torch.arange(2)
    mu, var = model.prior(2, y)

    assert mu.size() == (2, 3, 32, 32)
    assert var.size() == (2, 3, 32, 32)

    # Error
    with pytest.raises(ValueError):
        _ = model.prior(4, torch.arange(2))


def test_example_inverse(model) -> None:
    z = torch.randn(2, 3, 32, 32)
    x = model.inverse(z)

    assert x.size() == z.size()


def test_example_loss_func(model) -> None:
    x = torch.randn(2, 3, 32, 32)
    loss_dict = model.loss_func(x)

    assert isinstance(loss_dict, dict)
    assert loss_dict["loss"] != 0
    assert loss_dict["log_prob"] != 0
    assert loss_dict["logdet"] != 0
    assert loss_dict["classification"] == 0


def test_example_loss_func_conditional(model) -> None:
    x = torch.randn(2, 3, 32, 32)
    y = torch.arange(2)
    loss_dict = model.loss_func(x, y)

    assert isinstance(loss_dict, dict)
    assert loss_dict["loss"] != 0
    assert loss_dict["log_prob"] != 0
    assert loss_dict["logdet"] != 0
    assert loss_dict["classification"] != 0


def test_example_sample() -> None:
    model = flowlib.FlowModel(z_size=(3, 32, 32))
    model.flow_list = nn.ModuleList([
        TempLayer(), TempLayer()
    ])

    x = model.sample(5)
    assert x.size() == (5, 3, 32, 32)


def test_example_sample_conditional() -> None:
    model = flowlib.FlowModel(z_size=(3, 32, 32))
    model.flow_list = nn.ModuleList([
        TempLayer(), TempLayer()
    ])

    x = model.sample(5, torch.arange(5))
    assert x.size() == (5, 3, 32, 32)


def test_example_reconstruct(model) -> None:
    x = torch.randn(2, 3, 32, 32)
    recon = model.reconstruct(x)

    assert recon.size() == x.size()
