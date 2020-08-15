
import unittest

import torch

import flowlib


class TestSlicing(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.Slicing()

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), (4, 6, 8, 8))
        self.assertFalse(torch.isnan(z).any())
        self.assertTupleEqual(logdet.size(), (4,))

    def test_inverse(self):
        z = torch.randn(4, 4, 8, 8)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), (4, 2, 8, 8))
        self.assertFalse(torch.isnan(x).any())


class TestAbsSurjection(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.AbsSurjection()

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertFalse(torch.isnan(z).any())
        self.assertTupleEqual(logdet.size(), (4,))

    def test_inverse(self):
        z = torch.randn(4, 3, 8, 8)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())
        self.assertFalse(torch.isnan(x).any())


class TestMaxSurjection(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.MaxSurjection()

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertFalse(torch.isnan(z).any())
        self.assertTupleEqual(logdet.size(), (4,))

    def test_inverse(self):
        z = torch.randn(4, 3, 8, 8)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())
        self.assertFalse(torch.isnan(x).any())


class TestSortSurjection(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.SortSurjection()

    def test_forward(self):
        x = torch.randn(4, 3, 8, 8)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertFalse(torch.isnan(z).any())
        self.assertTupleEqual(logdet.size(), (4,))

    def test_inverse(self):
        z = torch.randn(4, 3, 8, 8)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())
        self.assertFalse(torch.isnan(x).any())


if __name__ == "__main__":
    unittest.main()
