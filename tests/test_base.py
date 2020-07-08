
import unittest

import torch
from torch import nn

import flowlib


class TestLinearZeros(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.LinearZeros(3, 4)

    def test_forward(self):
        x = torch.randn(4, 3)
        z = self.model(x)

        self.assertTupleEqual(z.size(), (4, 4))
        self.assertFalse(torch.isnan(z).any())


# Temporal layer class
class TempLayer(nn.Module):
    def forward(self, x):
        return x, x.abs().mean().unsqueeze(0)

    def inverse(self, z):
        return z


class TestFlowModel(unittest.TestCase):

    def setUp(self):
        self.model = flowlib.FlowModel()
        self.model.flow_list = nn.ModuleList([
            TempLayer(), TempLayer()
        ])

    def test_forward(self):
        x = torch.randn(2, 3, 32, 32)
        z, logdet = self.model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertTupleEqual(logdet.size(), (2,))

    def test_prior(self):
        mu, var = self.model.prior(2)

        self.assertTupleEqual(mu.size(), (2, 3, 32, 32))
        self.assertTupleEqual(var.size(), (2, 3, 32, 32))

        # Conditional
        y = torch.arange(2)
        mu, var = self.model.prior(2, y)

        self.assertTupleEqual(mu.size(), (2, 3, 32, 32))
        self.assertTupleEqual(var.size(), (2, 3, 32, 32))

        # Error
        with self.assertRaises(ValueError):
            _ = self.model.prior(4, torch.arange(2))

    def test_inverse(self):
        z = torch.randn(2, 3, 32, 32)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())

    def test_loss_func(self):
        x = torch.randn(2, 3, 32, 32)
        loss_dict = self.model.loss_func(x)

        self.assertIsInstance(loss_dict, dict)
        self.assertNotEqual(loss_dict["loss"], 0)
        self.assertNotEqual(loss_dict["log_prob"], 0)
        self.assertNotEqual(loss_dict["logdet"], 0)
        self.assertEqual(loss_dict["classification"], 0)

    def test_loss_func_conditional(self):
        x = torch.randn(2, 3, 32, 32)
        y = torch.arange(2)
        loss_dict = self.model.loss_func(x, y)

        self.assertIsInstance(loss_dict, dict)
        self.assertNotEqual(loss_dict["loss"], 0)
        self.assertNotEqual(loss_dict["log_prob"], 0)
        self.assertNotEqual(loss_dict["logdet"], 0)
        self.assertNotEqual(loss_dict["classification"], 0)

    def test_sample(self):
        model = flowlib.FlowModel(z_size=(3, 32, 32))
        model.flow_list = nn.ModuleList([
            TempLayer(), TempLayer()
        ])

        x = model.sample(5)
        self.assertTupleEqual(x.size(), (5, 3, 32, 32))

    def test_sample_conditional(self):
        model = flowlib.FlowModel(z_size=(3, 32, 32))
        model.flow_list = nn.ModuleList([
            TempLayer(), TempLayer()
        ])

        x = model.sample(5, torch.arange(5))
        self.assertTupleEqual(x.size(), (5, 3, 32, 32))

    def test_reconstruct(self):
        x = torch.randn(2, 3, 32, 32)
        recon = self.model.reconstruct(x)

        self.assertTupleEqual(recon.size(), x.size())


if __name__ == "__main__":
    unittest.main()
