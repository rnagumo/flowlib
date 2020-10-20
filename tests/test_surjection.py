import torch
import flowlib


def test_slicing_forward() -> None:
    model = flowlib.Slicing()
    x = torch.randn(4, 3, 8, 8)
    z, logdet = model(x)

    assert z.size() == (4, 6, 8, 8)
    assert not torch.isnan(z).any()
    assert logdet.size() == (4,)


def test_slicing_inverse() -> None:
    model = flowlib.Slicing()
    z = torch.randn(4, 4, 8, 8)
    x = model.inverse(z)

    assert x.size() == (4, 2, 8, 8)
    assert not torch.isnan(x).any()


def test_abs_surjection_forward() -> None:
    model = flowlib.AbsSurjection()
    x = torch.randn(4, 3, 8, 8)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert not torch.isnan(z).any()
    assert logdet.size() == (4,)


def test_abs_surjection_inverse() -> None:
    model = flowlib.AbsSurjection()
    z = torch.randn(4, 3, 8, 8)
    x = model.inverse(z)

    assert x.size() == z.size()
    assert not torch.isnan(x).any()


def test_max_surjection_forward() -> None:
    model = flowlib.MaxSurjection()
    x = torch.randn(4, 3, 8, 8)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert not torch.isnan(z).any()
    assert logdet.size() == (4,)


def test_max_surjection_inverse() -> None:
    model = flowlib.MaxSurjection()
    z = torch.randn(4, 3, 8, 8)
    x = model.inverse(z)

    assert x.size() == z.size()
    assert not torch.isnan(x).any()


def test_sort_surjection_forward() -> None:
    model = flowlib.SortSurjection()
    x = torch.randn(4, 3, 8, 8)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert not torch.isnan(z).any()
    assert logdet.size() == (4,)


def test_sort_surjection_inverse() -> None:
    model = flowlib.SortSurjection()
    z = torch.randn(4, 3, 8, 8)
    x = model.inverse(z)

    assert x.size() == z.size()
    assert not torch.isnan(x).any()
