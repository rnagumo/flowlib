
import unittest

import torch

import flowlib


class TestSqueeze(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.Squeeze()

    def test_forward(self):
        x = torch.arange(1, 17).view(1, 1, 4, 4)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), (1, 4, 2, 2))
        self.assertTupleEqual(logdet.size(), ())

        z_true = torch.tensor([[
            [[1, 3], [9, 11]],
            [[2, 4], [10, 12]],
            [[5, 7], [13, 15]],
            [[6, 8], [14, 16]],
        ]])

        self.assertTrue((z == z_true).all())

    def test_inverse(self):
        z = torch.tensor([[
            [[1, 3], [9, 11]],
            [[2, 4], [10, 12]],
            [[5, 7], [13, 15]],
            [[6, 8], [14, 16]],
        ]])
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), (1, 1, 4, 4))
        self.assertTrue((x == torch.arange(1, 17).view(1, 1, 4, 4)).all())


class TestUnsqueeze(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.Unsqueeze()

    def test_forward(self):
        x = torch.tensor([[
            [[1, 3], [9, 11]],
            [[2, 4], [10, 12]],
            [[5, 7], [13, 15]],
            [[6, 8], [14, 16]],
        ]])
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), (1, 1, 4, 4))
        self.assertTupleEqual(logdet.size(), ())

        self.assertTrue((z == torch.arange(1, 17).view(1, 1, 4, 4)).all())

    def test_inverse(self):
        z = torch.arange(1, 17).view(1, 1, 4, 4)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), (1, 4, 2, 2))

        x_true = torch.tensor([[
            [[1, 3], [9, 11]],
            [[2, 4], [10, 12]],
            [[5, 7], [13, 15]],
            [[6, 8], [14, 16]],
        ]])

        self.assertTrue((x == x_true).all())


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.Preprocess()

    def test_forward(self):
        x = torch.rand(4, 3, 8, 8)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertTupleEqual(logdet.size(), ())
        self.assertFalse(torch.isnan(z).any())

    def test_inverse(self):
        z = torch.randn(4, 3, 8, 8) * 100
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())
        self.assertFalse(torch.isnan(x).any())
        self.assertTrue((x >= 0).all() and (x <= 1).all())


if __name__ == "__main__":
    unittest.main()
