
import unittest

import torch

import flowlib


class TestGlow(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.Glow()

    def test_forward(self):
        x = torch.rand(2, 3, 32, 32)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), (2, 48, 4, 4))
        self.assertFalse(torch.isnan(z).any())
        self.assertTupleEqual(logdet.size(), (2,))

    def test_inverse(self):
        # Initialize actnorm by forward step
        x = torch.rand(2, 3, 32, 32)
        self.model(x)

        z = torch.randn(2, 48, 4, 4)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), (2, 3, 32, 32))
        self.assertFalse(torch.isnan(x).any())

    def test_loss_func(self):
        x = torch.rand(2, 3, 32, 32)
        loss_dict = self.model.loss_func(x)

        self.assertIsInstance(loss_dict, dict)
        self.assertGreater(loss_dict["loss"], 0)
        self.assertGreater(loss_dict["log_prob"], 0)
        self.assertTrue(loss_dict["logdet"] < 0 or loss_dict["logdet"] >= 0)

    def test_sample(self):
        # Initialize actnorm by forward step
        x = torch.rand(2, 3, 32, 32)
        self.model(x)

        x = self.model.sample(5)
        self.assertTupleEqual(x.size(), (5, 3, 32, 32))
        self.assertFalse(torch.isnan(x).any())

    def test_inference_with_other_shape(self):
        model = flowlib.Glow(3, 64, image_size=64, depth=12, level=2)
        x = torch.rand(2, 3, 64, 64)
        z, logdet = model(x)

        self.assertTupleEqual(z.size(), (2, 24, 16, 16))
        self.assertFalse(torch.isnan(z).any())
        self.assertTupleEqual(logdet.size(), (2,))


if __name__ == "__main__":
    unittest.main()
