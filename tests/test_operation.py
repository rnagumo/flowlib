import torch
import flowlib


def test_squeeze_forward() -> None:
    model = flowlib.Squeeze()
    x = torch.arange(1, 17).view(1, 1, 4, 4)
    z, logdet = model(x)

    assert z.size() == (1, 4, 2, 2)
    assert logdet.size() == (1,)

    z_true = torch.tensor(
        [
            [
                [[1, 3], [9, 11]],
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
            ]
        ]
    )
    assert (z == z_true).all()


def test_squeeze_inverse() -> None:
    model = flowlib.Squeeze()
    z = torch.tensor(
        [
            [
                [[1, 3], [9, 11]],
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
            ]
        ]
    )
    x = model.inverse(z)

    assert x.size() == (1, 1, 4, 4)
    assert (x == torch.arange(1, 17).view(1, 1, 4, 4)).all()


def test_unsqueeze_forward() -> None:
    model = flowlib.Unsqueeze()
    x = torch.tensor(
        [
            [
                [[1, 3], [9, 11]],
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
            ]
        ]
    )
    z, logdet = model(x)

    assert z.size() == (1, 1, 4, 4)
    assert logdet.size() == (1,)
    assert (z == torch.arange(1, 17).view(1, 1, 4, 4)).all()


def test_unsqueeze_inverse() -> None:
    model = flowlib.Unsqueeze()
    z = torch.arange(1, 17).view(1, 1, 4, 4)
    x = model.inverse(z)

    assert x.size() == (1, 4, 2, 2)

    x_true = torch.tensor(
        [
            [
                [[1, 3], [9, 11]],
                [[2, 4], [10, 12]],
                [[5, 7], [13, 15]],
                [[6, 8], [14, 16]],
            ]
        ]
    )
    assert (x == x_true).all()


def test_channelwise_split_forward() -> None:
    model = flowlib.ChannelwiseSplit(4)
    x = torch.rand(4, 4, 8, 8)
    z, logdet = model(x)

    assert z.size() == (4, 2, 8, 8)
    assert not torch.isnan(z).any()
    assert logdet.size() == (4,)


def test_channelwise_split_inverse() -> None:
    model = flowlib.ChannelwiseSplit(4)
    z = torch.randn(4, 2, 8, 8)
    x = model.inverse(z)

    assert x.size() == (4, 4, 8, 8)
    assert not torch.isnan(x).any()


def test_preprocess_forward() -> None:
    model = flowlib.Preprocess()
    x = torch.rand(4, 3, 8, 8)
    z, logdet = model(x)

    assert z.size() == x.size()
    assert logdet.size() == (4,)
    assert not torch.isnan(z).any()


def test_preprocess_inverse() -> None:
    model = flowlib.Preprocess()
    z = torch.randn(4, 3, 8, 8) * 100
    x = model.inverse(z)

    assert x.size() == z.size()
    assert not torch.isnan(x).any()
    assert (x >= 0).all() and (x <= 1).all()
