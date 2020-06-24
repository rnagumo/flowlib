
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
        x = torch.rand(2, 3, 32, 32)
        z = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertFalse(torch.isnan(z).any())

    def test_loss_func(self):
        x = torch.rand(2, 3, 32, 32)
        loss_dict = self.model.loss_func(x)

        self.assertIsInstance(loss_dict, dict)
        self.assertGreater(loss_dict["loss"], 0)

    def test_inference(self):
        x = torch.rand(2, 3, 32, 32)
        z, logdet = self.model.inference(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertFalse(torch.isnan(z).any())
        self.assertTupleEqual(logdet.size(), ())

    def test_inverse(self):
        z = torch.rand(2, 3, 32, 32)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())
        self.assertFalse(torch.isnan(x).any())

    def test_sample(self):
        x = self.model.sample(5)
        self.assertTupleEqual(x.size(), (5, 3, 32, 32))
        self.assertFalse(torch.isnan(x).any())

    def test_other_shape(self):
        model = flowlib.Glow(3, 64, image_size=64)
        x = torch.rand(2, 3, 64, 64)
        z, logdet = model.inference(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertFalse(torch.isnan(z).any())
        self.assertTupleEqual(logdet.size(), ())


if __name__ == "__main__":
    unittest.main()
