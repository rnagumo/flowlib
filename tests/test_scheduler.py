import warnings

import torch
from torch import nn, optim

import flowlib


def test_schedule() -> None:
    model = nn.Sequential(nn.Linear(3, 3))
    optimizer = optim.Adam(model.parameters())
    scheduler = flowlib.NoamScheduler(optimizer, 50, 1)

    results_list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(100):
            scheduler.step()
            results_list.append(scheduler.get_last_lr()[0])

    results = torch.tensor(results_list)

    # Check linear increment
    assert (results[1:] - results[:-1] > 0)[:49].all()

    # Check decay
    assert (results[1:] - results[:-1] < 0)[49:].all()
