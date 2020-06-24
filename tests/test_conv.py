
import unittest

import torch

import flowlib


class TestInvertibleConv(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.InvertibleConv(in_channel=3)

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertTupleEqual(logdet.size(), ())

    def test_inverse(self):
        z = torch.randn(4, 3, 8, 8)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())


class TestInvertibleConvLU(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.InvertibleConvLU(in_channel=3)

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertTupleEqual(logdet.size(), ())

    def test_inverse(self):
        z = torch.randn(4, 3, 8, 8)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())


if __name__ == "__main__":
    unittest.main()
