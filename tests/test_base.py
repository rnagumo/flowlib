
import unittest

import torch
from torch import nn

import flowlib


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

    def test_init(self):
        model = flowlib.FlowModel(in_size=(3, 4, 4))
        model.flow_list = nn.ModuleList([
            TempLayer(), TempLayer()
        ])

        self.assertTupleEqual(model._prior_mu.size(), (3, 4, 4))
        self.assertTupleEqual(model._prior_var.size(), (3, 4, 4))

    def test_forward(self):
        x = torch.randn(2, 3, 4, 4)
        z = self.model(x)

        self.assertTupleEqual(z.size(), x.size())

    def test_loss_func(self):
        x = torch.randn(2, 3, 4, 4)
        loss_dict = self.model.loss_func(x)

        self.assertIsInstance(loss_dict, dict)
        self.assertGreater(loss_dict["loss"], 0)
        self.assertGreater(loss_dict["log_prob"], 0)
        self.assertTrue(loss_dict["logdet"] < 0 or loss_dict["logdet"] >= 0)

    def test_inference(self):
        x = torch.randn(2, 3, 4, 4)
        z, logdet = self.model.inference(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertTupleEqual(logdet.size(), (2,))

    def test_inverse(self):
        z = torch.randn(2, 3, 4, 4)
        x = self.model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())

    def test_sample(self):
        model = flowlib.FlowModel(in_size=(3, 4, 4))
        model.flow_list = nn.ModuleList([
            TempLayer(), TempLayer()
        ])

        x = model.sample(5)
        self.assertTupleEqual(x.size(), (5, 3, 4, 4))

    def test_reconstruct(self):
        x = torch.randn(2, 3, 4, 4)
        recon = self.model.reconstruct(x)

        self.assertTupleEqual(recon.size(), x.size())


if __name__ == "__main__":
    unittest.main()
