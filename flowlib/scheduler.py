from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """Noam LR schaduler.

    Noam scheduler is proposed in [Vaswani+, 2017] to train Transformer. "Noam" is the name of the
    second author.

    ref) Vaswani+, 2017. "Attention Is All You Need."
    https://arxiv.org/abs/1706.03762

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        warmup_steps (int): Number of warm-up steps to increase lr linearly.
        d_dim (int): Dimension size of output.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, d_dim: int) -> None:

        self.warmup_steps = warmup_steps
        self.d_dim = d_dim

        # For mypy
        self.last_epoch: int
        self.base_lrs: List

        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Custom learning rate."""

        if self.last_epoch == 0:
            lr = self.base_lrs
        else:
            lr = [
                (
                    base_lr
                    * self.d_dim ** -0.5
                    * min(self.last_epoch ** -0.5, self.last_epoch * self.warmup_steps ** -1.5)
                )
                for base_lr in self.base_lrs
            ]

        return lr
