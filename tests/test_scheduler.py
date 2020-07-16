
import unittest
import warnings

import torch
from torch import nn, optim

import flowlib


class TestNoamScheduler(unittest.TestCase):

    def test_schedule(self):
        model = nn.Sequential(nn.Linear(3, 3))
        optimizer = optim.Adam(model.parameters())
        scheduler = flowlib.NoamScheduler(optimizer, 50, 1)

        results = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(100):
                scheduler.step()
                results.append(scheduler.get_last_lr()[0])

        results = torch.tensor(results)

        # Check linear increment
        self.assertTrue((results[1:] - results[:-1] > 0)[:49].all())

        # Check decay
        self.assertTrue((results[1:] - results[:-1] < 0)[49:].all())


if __name__ == "__main__":
    unittest.main()
