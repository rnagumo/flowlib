
import unittest

import torch

import flowlib


class TestGlow(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.Glow()

    def test_init(self):
        self.assertTupleEqual(self.model._prior_mu.size(), (3, 32, 32))
        self.assertTupleEqual(self.model._prior_var.size(), (3, 32, 32))

    def test_forward(self):
        x = torch.randn(2, 3, 32, 32)
        z = self.model(x)

        self.assertTupleEqual(z.size(), x.size())

    def test_loss_func(self):
        x = torch.randn(2, 3, 32, 32)
        loss_dict = self.model.loss_func(x)

        self.assertIsInstance(loss_dict, dict)
        self.assertGreater(loss_dict["loss"], 0)

    def test_inference(self):
        x = torch.randn(2, 3, 32, 32)
        z, logdet = self.model.inference(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertTupleEqual(logdet.size(), ())

    def test_inverse(self):
        z = torch.randn(2, 3, 32, 32)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())

    def test_sample(self):
        x = self.model.sample(5)
        self.assertTupleEqual(x.size(), (5, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
