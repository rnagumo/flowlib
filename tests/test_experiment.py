from typing import Tuple

from copy import deepcopy
import re
import tempfile

import torch
from torch import nn, Tensor

import flowlib
from flowlib.base import FlowLayer, FlowModel


def test_trainer_run() -> None:

    model = TempModel()
    train_data = TempDataset()
    test_data = TempDataset()

    org_params = deepcopy(model.state_dict())

    with tempfile.TemporaryDirectory() as logdir:
        trainer = flowlib.Trainer(logdir=logdir)
        trainer.run(model, train_data, test_data)

        root = trainer._logdir
        assert (root / "training.log").exists()
        assert (root / "config.json").exists()

    updated_params = model.state_dict()
    for key in updated_params:
        if (
            key != "buffer"
            and re.match("^y_prior", key) is None
            and re.match("^y_projector", key) is None
        ):
            assert not (updated_params[key] == org_params[key]).all()


class TempLayer(FlowLayer):
    def __init__(self) -> None:
        super().__init__()

        self.layer = nn.Linear(3 * 32 * 32, 3 * 32 * 32, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        x = x.view(-1, 3 * 32 * 32)
        x = self.layer(x)
        x = x.view(-1, 3, 32, 32)

        logdet = x.sum()

        return x, logdet

    def inverse(self, z: Tensor) -> Tensor:

        z = z.view(-1, 3 * 32 * 32)
        z = self.layer(z)
        z = z.view(-1, 3, 32, 32)

        return z


class TempModel(FlowModel):
    def __init__(self) -> None:
        super().__init__()

        self.flow_list = nn.ModuleList(
            [
                TempLayer(),
                TempLayer(),
            ]
        )


class TempDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self._data = torch.rand(10, 3, 32, 32)
        self._label = torch.randint(0, 10, (10,))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self._data[index], self._label[index]

    def __len__(self) -> int:
        return self._data.size(0)
