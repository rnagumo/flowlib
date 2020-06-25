
import unittest

import torch

import flowlib


class TestActNorm2d(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.ActNorm2d(in_channels=3)

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertFalse(torch.isnan(z).any())
        self.assertTupleEqual(logdet.size(), (1,))

    def test_inverse(self):
        z = torch.randn(4, 3, 8, 8)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())
        self.assertFalse(torch.isnan(x).any())


if __name__ == "__main__":
    unittest.main()
