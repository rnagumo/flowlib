
"""Trainer class."""

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
from torch import optim
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

    def __init__(self, model: flowlib.FlowModel, config: dict):
        # Params
        self.model = model
        self.config = Config(**config)

        # Attributes
        self.logdir: pathlib.Path
        self.logger: logging.Logger
        self.writer: tb.SummaryWriter
        self.train_loader: dataloader.DataLoader
        self.test_loader: dataloader.DataLoader
        self.optimizer: optimizer.Optimizer
        self.scheduler: optim.lr_scheduler._LRScheduler
        self.device: torch.device
        self.pbar: tqdm.tqdm

        # Training utils
        self.global_steps = 0
        self.postfix: Dict[str, float] = {}

    def check_logdir(self) -> None:
        """Checks log directory.

        This method specifies logdir and make the directory if it does not
        exist.
        """

        self.logdir = pathlib.Path(
            self.config.logdir, time.strftime("%Y%m%d%H%M"))
        self.logdir.mkdir(parents=True, exist_ok=True)

    def init_logger(self, save_file: bool = True) -> None:
        """Initalizes logger.

        Args:
            save_file (bool, optoinal): If `True`, save log file.
        """

        # Initialize logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Set stream handler (console)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh_fmt = logging.Formatter("%(asctime)s - %(module)s.%(funcName)s "
                                   "- %(levelname)s : %(message)s")
        sh.setFormatter(sh_fmt)
        logger.addHandler(sh)

        # Set file handler (log file)
        if save_file:
            fh = logging.FileHandler(filename=self.logdir / "training.log")
            fh.setLevel(logging.DEBUG)
            fh_fmt = logging.Formatter("%(asctime)s - %(module)s.%(funcName)s "
                                       "- %(levelname)s : %(message)s")
            fh.setFormatter(fh_fmt)
            logger.addHandler(fh)

        self.logger = logger

    def init_writer(self) -> None:
        """Initializes tensorboard writer."""

        self.writer = tb.SummaryWriter(str(self.logdir))

    def load_dataloader(self) -> None:
        """Loads data loader for training and test."""

        self.logger.info("Load dataset")

        # Dataset
        if self.config.dataset_name == "cifar":
            # Transform
            # For training, augment datasets with horizontal flips according to
            # Real-NVP (L. Dinh+, 2017) paper.
            trans_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            trans_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            # Kwargs for dataset
            train_kwargs = {"root": self.config.data_dir, "download": True,
                            "train": True, "transform": trans_train}
            test_kwargs = {"root": self.config.data_dir, "download": True,
                           "train": False, "transform": trans_test}

            train_data = datasets.CIFAR10(**train_kwargs)
            test_data = datasets.CIFAR10(**test_kwargs)

        elif self.config.dataset_name == "celeba":
            # Transform
            transform = transforms.Compose([
                transforms.CenterCrop(self.config.image_size),
                transforms.ToTensor(),
            ])

            # Kwargs for dataset
            train_kwargs = {"root": self.config.data_dir, "download": True,
                            "split": "train", "transform": transform}
            test_kwargs = {"root": self.config.data_dir, "download": True,
                           "split": "test", "transform": transform}

            train_data = datasets.CelebA(**train_kwargs)
            test_data = datasets.CelebA(**test_kwargs)
        else:
            raise ValueError(
                f"Unexpected dataset name: {self.config.dataset_name}")

        # Params for GPU
        if torch.cuda.is_available():
            kwargs = {"num_workers": 0, "pin_memory": True}
        else:
            kwargs = {}

        self.train_loader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=self.config.batch_size,
            **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            test_data, shuffle=False, batch_size=self.config.batch_size,
            **kwargs)

        self.logger.info(f"Train dataset size: {len(self.train_loader)}")
        self.logger.info(f"Test dataset size: {len(self.test_loader)}")

    def train(self) -> None:
        """Trains model."""

        for data, label in self.train_loader:
            self.model.train()

            # Data to device
            data = data.to(self.device)
            label = (label.to(self.device) if self.config.y_conditional
                     else None)

            # Forward
            self.optimizer.zero_grad()
            loss_dict = self.model.loss_func(data, label)
            loss = loss_dict["loss"].mean()

            # Backward and update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm)
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.config.max_grad_value)
            self.optimizer.step()
            self.scheduler.step()

            # Progress bar update
            self.global_steps += 1
            self.pbar.update(1)

            self.postfix["train/loss"] = loss.item()
            self.pbar.set_postfix(self.postfix)

            # Summary
            for key, value in loss_dict.items():
                self.writer.add_scalar(
                    f"train/{key}", value.mean(), self.global_steps)

            # Test
            if self.global_steps % self.config.test_interval == 0:
                self.test()

            # Save checkpoint
            if self.global_steps % self.config.save_interval == 0:
                self.save_checkpoint()

                loss_logger = {k: v.mean() for k, v in loss_dict.items()}
                self.logger.debug(
                    f"Train loss (steps={self.global_steps}): "
                    f"{loss_logger}")

                self.save_plots()

            # Check step limit
            if self.global_steps >= self.config.max_steps:
                break

    def test(self) -> None:
        """Tests model."""

        # Logger for loss
        loss_logger: DefaultDict[str, float] = collections.defaultdict(float)

        # Run
        self.model.eval()
        for data, label in self.test_loader:
            with torch.no_grad():
                # Data to device
                data = data.to(self.device)
                label = (label.to(self.device) if self.config.y_conditional
                         else None)

                # Calculate loss
                loss_dict = self.model.loss_func(data, label)
                loss = loss_dict["loss"]

            # Update progress bar
            self.postfix["test/loss"] = loss.mean().item()
            self.pbar.set_postfix(self.postfix)

            # Save loss
            for key, value in loss_dict.items():
                loss_logger[key] += value.sum().item()

        # Summary
        for key, value in loss_logger.items():
            self.writer.add_scalar(
                f"test/{key}", value / (len(self.test_loader)),
                self.global_steps)

        self.logger.debug(
            f"Test loss (steps={self.global_steps}): {loss_logger}")

    def save_checkpoint(self) -> None:
        """Saves trained model and optimizer to checkpoint file.

        Args:
            loss (float): Saved loss value.
        """

        # Log
        self.logger.debug("Save trained model")

        # Remove unnecessary prefix from state dict keys
        model_state_dict = {}
        for k, v in self.model.state_dict().items():
            model_state_dict[k.replace("module.", "")] = v

        optimizer_state_dict = {}
        for k, v in self.optimizer.state_dict().items():
            optimizer_state_dict[k.replace("module.", "")] = v

        # Save model
        state_dict = {
            "steps": self.global_steps,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        path = self.logdir / f"checkpoint_{self.global_steps}.pt"
        torch.save(state_dict, path)

    def save_configs(self) -> None:
        """Saves setting including config and args in json format."""

        self.logger.debug("Save configs")

        config = dataclasses.asdict(self.config)
        config["logdir"] = str(self.logdir)

        with (self.logdir / "config.json").open("w") as f:
            json.dump(config, f)

    def save_plots(self) -> None:
        """Save reconstructed and sampled plots."""

        def gridshow(img):
            grid = make_grid(img)
            npgrid = grid.permute(1, 2, 0).numpy()
            plt.imshow(npgrid, interpolation="nearest")

        with torch.no_grad():
            x, label = next(iter(self.test_loader))
            x = x[:16].to(self.device)
            label = (label[:16].to(self.device) if self.config.y_conditional
                     else None)

            recon = self.model.reconstruct(x)
            sample = self.model.sample(16, label)

        x = x.cpu()
        recon = recon.cpu()
        sample = sample.cpu()

        # Plot
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
        plt.savefig(self.logdir / f"fig_{self.global_steps}.png")
        plt.close()

    def quit(self) -> None:
        """Post process."""

        self.logger.info("Quit base run method")
        self.save_configs()
        self.writer.close()

    def _base_run(self) -> None:
        """Base running method."""

        self.logger.info("Start experiment")

        # Device
        if self.config.gpus:
            self.device = torch.device(f"cuda:{self.config.gpus}")
        else:
            self.device = torch.device("cpu")

        # Data
        self.load_dataloader()

        # Model
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), **self.config.optimizer_params)
        self.scheduler = flowlib.NoamScheduler(
            self.optimizer, **self.config.scheduler_params)

        # Progress bar
        self.pbar = tqdm.tqdm(total=self.config.max_steps)
        self.global_steps = 0
        self.postfix = {"train/loss": 0.0, "test/loss": 0.0}

        # Run training
        while self.global_steps < self.config.max_steps:
            self.train()

        self.pbar.close()
        self.logger.info("Finish training")

    def run(self) -> None:
        """Main run method."""

        # Settings
        self.check_logdir()
        self.init_logger()
        self.init_writer()

        self.logger.info("Start run")
        self.logger.info(f"Logdir: {self.logdir}")
        self.logger.info(f"Params: {self.config}")

        # Run
        try:
            self._base_run()
        except Exception as e:
            self.logger.exception(f"Run function error: {e}")
        finally:
            self.quit()

        self.logger.info("Finish run")
