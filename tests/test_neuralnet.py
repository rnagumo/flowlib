
import unittest

import torch

import flowlib


class TestConv2dZeros(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.Conv2dZeros(3, 3, 3, padding=1)

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertFalse(torch.isnan(z).any())


class TestResidualBlock(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.ResidualBlock(3, 4, 3)

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertFalse(torch.isnan(z).any())


if __name__ == "__main__":
    unittest.main()
