
import unittest

import torch

import flowlib


class TempNet(torch.nn.Module):
    def forward(self, x):
        return x, x


class TestAffineCoupling(unittest.TestCase):

    def base_case(self, mask_type, inverse_mask):
        scale_trans_net = TempNet()
        model = flowlib.AffineCoupling(
            scale_trans_net, mask_type=mask_type, inverse_mask=inverse_mask)

        # Forward
        x = torch.randn(4, 3, 8, 8)
        z, logdet = model(x)

        self.assertTupleEqual(z.size(), x.size())
        self.assertTupleEqual(logdet.size(), ())

        # Inverse
        z = torch.randn(4, 3, 8, 8)
        x = model.inverse(z)

        self.assertTupleEqual(x.size(), z.size())

    def test_checkerboard_true(self):
        self.base_case("checkerboard", True)

    def test_checkerboard_false(self):
        self.base_case("checkerboard", False)

    def test_channel_wise_true(self):
        self.base_case("channel_wise", True)

    def test_channel_wise_false(self):
        self.base_case("channel_wise", False)


class TestMask(unittest.TestCase):

    def test_checkerboard_mask(self):

        mask = flowlib.checkerboard_mask(3, 4, inverse=False)
        true_mask = torch.tensor([[1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [1, 0, 1, 0]])
        self.assertTrue((mask == true_mask).all())

        mask = flowlib.checkerboard_mask(3, 4, inverse=True)
        true_mask = torch.tensor([[0, 1, 0, 1],
                                  [1, 0, 1, 0],
                                  [0, 1, 0, 1]])
        self.assertTrue((mask == true_mask).all())

    def test_channel_wise_mask(self):

        mask = flowlib.channel_wise_mask(6, inverse=False)
        self.assertTrue((mask[:3] == 1).all())
        self.assertTrue((mask[3:] == 0).all())

        mask = flowlib.channel_wise_mask(6, inverse=True)
        self.assertTrue((mask[:3] == 0).all())
        self.assertTrue((mask[3:] == 1).all())


if __name__ == "__main__":
    unittest.main()
