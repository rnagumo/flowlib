from typing import Dict, DefaultDict, Union, Optional

import collections
import dataclasses
import json
import logging
import pathlib
import time

import matplotlib.pyplot as plt
import tqdm

import torch
from torch import Tensor, optim
from torch.optim import optimizer
from torch.utils.data import dataloader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import tensorboardX as tb

import flowlib


@dataclasses.dataclass
class Config:
    # From kwargs
    cuda: str
    model: str
    seed: int
    batch_size: int
    max_steps: int
    test_interval: int
    save_interval: int
    y_conditional: bool

    # From config
    glow_params: dict
    optimizer_params: dict
    scheduler_params: dict
    max_grad_value: float
    max_grad_norm: float
    image_size: int

    # From params
    logdir: Union[str, pathlib.Path]
    gpus: Optional[str]
    data_dir: Union[str, pathlib.Path]
    dataset_name: str


class Trainer:
    """Trainer class for ML models.

    Args:
        model (flowlib.FlowModel): ML model.
        config (dict): Dictionary of hyper-parameters.
    """

    def __init__(self, model: flowlib.FlowModel, config: dict) -> None:

        self._model = model
        self._config = Config(**config)
        self._global_steps = 0
        self._postfix: Dict[str, float] = {}

        self._logdir: pathlib.Path
        self._logger: logging.Logger
        self._writer: tb.SummaryWriter
        self._train_loader: dataloader.DataLoader
        self._test_loader: dataloader.DataLoader
        self._optimizer: optimizer.Optimizer
        self._scheduler: optim.lr_scheduler._LRScheduler
        self._device: torch.device
        self._pbar: tqdm.tqdm

    def run(self) -> None:
        """Run main method."""

        self._make_logdir()
        self._init_logger()
        self._init_writer()

        try:
            self._run_body()
        except Exception as e:
            self._logger.exception(f"Run function error: {e}")
        finally:
            self._quit()

    def _run_body(self) -> None:

        self._logger.info("Start experiment")
        self._logger.info(f"Logdir: {self._logdir}")
        self._logger.info(f"Params: {self._config}")

        if self._config.gpus:
            self._device = torch.device(f"cuda:{self._config.gpus}")
        else:
            self._device = torch.device("cpu")

        self._load_dataloader()
        self._model = self._model.to(self._device)
        self._optimizer = optim.Adam(self._model.parameters(), **self._config.optimizer_params)
        self._scheduler = flowlib.NoamScheduler(self._optimizer, **self._config.scheduler_params)

        self._pbar = tqdm.tqdm(total=self._config.max_steps)
        self._global_steps = 0
        self._postfix = {"train/loss": 0.0, "test/loss": 0.0}

        while self._global_steps < self._config.max_steps:
            self._train()

        self._pbar.close()
        self._logger.info("Finish training")

    def _make_logdir(self) -> None:

        self._logdir = pathlib.Path(self._config.logdir, time.strftime("%Y%m%d%H%M"))
        self._logdir.mkdir(parents=True, exist_ok=True)

    def _init_logger(self) -> None:

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh_fmt = logging.Formatter(
            "%(asctime)s - %(module)s.%(funcName)s " "- %(levelname)s : %(message)s"
        )
        sh.setFormatter(sh_fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(filename=self._logdir / "training.log")
        fh.setLevel(logging.DEBUG)
        fh_fmt = logging.Formatter(
            "%(asctime)s - %(module)s.%(funcName)s " "- %(levelname)s : %(message)s"
        )
        fh.setFormatter(fh_fmt)
        logger.addHandler(fh)

        self._logger = logger

    def _init_writer(self) -> None:

        self._writer = tb.SummaryWriter(str(self._logdir))

    def _load_dataloader(self) -> None:

        self._logger.info("Load dataset")

        if self._config.dataset_name == "cifar":
            # Transform
            # For training, augment datasets with horizontal flips according to Real-NVP
            # (L. Dinh+, 2017) paper.
            trans_train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            trans_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

            train_kwargs = {
                "root": self._config.data_dir,
                "download": True,
                "train": True,
                "transform": trans_train,
            }
            test_kwargs = {
                "root": self._config.data_dir,
                "download": True,
                "train": False,
                "transform": trans_test,
            }

            train_data = datasets.CIFAR10(**train_kwargs)
            test_data = datasets.CIFAR10(**test_kwargs)

        elif self._config.dataset_name == "celeba":
            transform = transforms.Compose(
                [
                    transforms.CenterCrop(self._config.image_size),
                    transforms.ToTensor(),
                ]
            )

            train_kwargs = {
                "root": self._config.data_dir,
                "download": True,
                "split": "train",
                "transform": transform,
            }
            test_kwargs = {
                "root": self._config.data_dir,
                "download": True,
                "split": "test",
                "transform": transform,
            }

            train_data = datasets.CelebA(**train_kwargs)
            test_data = datasets.CelebA(**test_kwargs)
        else:
            raise ValueError(f"Unexpected dataset name: {self._config.dataset_name}")

        if torch.cuda.is_available():
            kwargs = {"num_workers": 0, "pin_memory": True}
        else:
            kwargs = {}

        self._train_loader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=self._config.batch_size, **kwargs
        )

        self._test_loader = torch.utils.data.DataLoader(
            test_data, shuffle=False, batch_size=self._config.batch_size, **kwargs
        )

        self._logger.info(f"Train dataset size: {len(self._train_loader)}")
        self._logger.info(f"Test dataset size: {len(self._test_loader)}")

    def _train(self) -> None:

        for data, label in self._train_loader:
            self._model.train()
            data = data.to(self._device)
            label = label.to(self._device) if self._config.y_conditional else None

            self._optimizer.zero_grad()
            loss_dict = self._model.loss_func(data, label)
            loss = loss_dict["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.max_grad_norm)
            torch.nn.utils.clip_grad_value_(self._model.parameters(), self._config.max_grad_value)
            self._optimizer.step()
            self._scheduler.step()

            self._global_steps += 1
            self._pbar.update(1)

            self._postfix["train/loss"] = loss.item()
            self._pbar.set_postfix(self._postfix)

            for key, value in loss_dict.items():
                self._writer.add_scalar(f"train/{key}", value.mean(), self._global_steps)

            if self._global_steps % self._config.test_interval == 0:
                self._test()

            if self._global_steps % self._config.save_interval == 0:
                self._save_checkpoint()

                loss_logger = {k: v.mean() for k, v in loss_dict.items()}
                self._logger.debug(f"Train loss (steps={self._global_steps}): " f"{loss_logger}")

                self._save_plots()

            if self._global_steps >= self._config.max_steps:
                break

    def _test(self) -> None:

        loss_logger: DefaultDict[str, float] = collections.defaultdict(float)
        self._model.eval()
        for data, label in self._test_loader:
            with torch.no_grad():
                data = data.to(self._device)
                label = label.to(self._device) if self._config.y_conditional else None

                loss_dict = self._model.loss_func(data, label)
                loss = loss_dict["loss"]

            self._postfix["test/loss"] = loss.mean().item()
            self._pbar.set_postfix(self._postfix)

            for key, value in loss_dict.items():
                loss_logger[key] += value.sum().item()

        for key, value in loss_logger.items():
            self._writer.add_scalar(
                f"test/{key}", value / (len(self._test_loader)), self._global_steps
            )

        self._logger.debug(f"Test loss (steps={self._global_steps}): {loss_logger}")

    def _save_checkpoint(self) -> None:

        self._logger.debug("Save trained model")

        # Remove unnecessary prefix from state dict keys
        model_state_dict = {}
        for k, v in self._model.state_dict().items():
            model_state_dict[k.replace("module.", "")] = v

        optimizer_state_dict = {}
        for k, v in self._optimizer.state_dict().items():
            optimizer_state_dict[k.replace("module.", "")] = v

        state_dict = {
            "steps": self._global_steps,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        path = self._logdir / f"checkpoint_{self._global_steps}.pt"
        torch.save(state_dict, path)

    def _save_configs(self) -> None:

        self._logger.debug("Save configs")
        config = dataclasses.asdict(self._config)
        config["logdir"] = str(self._logdir)

        with (self._logdir / "config.json").open("w") as f:
            json.dump(config, f)

    def _save_plots(self) -> None:
        def gridshow(img: Tensor) -> None:
            grid = make_grid(img)
            npgrid = grid.permute(1, 2, 0).numpy()
            plt.imshow(npgrid, interpolation="nearest")

        with torch.no_grad():
            x, label = next(iter(self._test_loader))
            x = x[:16].to(self._device)
            label = label[:16].to(self._device) if self._config.y_conditional else None

            recon = self._model.reconstruct(x)
            sample = self._model.sample(16, label)

            x = x.cpu()
            recon = recon.cpu()
            sample = sample.cpu()

        plt.figure(figsize=(20, 12))

        plt.subplot(311)
        gridshow(x)
        plt.title("Original")

        plt.subplot(312)
        gridshow(recon)
        plt.title("Reconstructed")

        plt.subplot(313)
        gridshow(sample)
        plt.title("Sampled")

        plt.tight_layout()
        plt.savefig(self._logdir / f"fig_{self._global_steps}.png")
        plt.close()

    def _quit(self) -> None:

        self._logger.info("Quit base run method")
        self._save_configs()
        self._writer.close()
