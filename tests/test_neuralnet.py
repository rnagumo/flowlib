
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


class TestConvBlock(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.ConvBlock(3, 4)

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z = self.model(x)

        self.assertTupleEqual(z.size(), (4, 6, 8, 8))
        self.assertFalse(torch.isnan(z).any())


if __name__ == "__main__":
    unittest.main()
