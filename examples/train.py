from typing import Tuple

import argparse
import json
import os
import pathlib
import random

import torch
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

import flowlib


def main() -> None:

    args = init_args()

    config_path = pathlib.Path(os.getenv("CONFIG_PATH", "./examples/config.json"))
    with config_path.open() as f:
        config = json.load(f)

    logdir = str(pathlib.Path(os.getenv("LOGDIR", "./logs/"), os.getenv("EXPERIMENT_NAME", "tmp")))
    dataset_name = os.getenv("DATASET_NAME", "cifar")
    data_dir = pathlib.Path(os.getenv("DATASET_DIR", "./data/"), dataset_name)

    params = vars(args)
    args_seed = params.pop("seed")
    args_cuda = params.pop("cuda")
    args_model = params.pop("model")

    torch.manual_seed(args_seed)
    random.seed(args_seed)

    use_cuda = torch.cuda.is_available() and args_cuda != "null"
    gpus = args_cuda if use_cuda else ""

    params.update(
        {
            "logdir": str(logdir),
            "gpus": gpus,
        }
    )

    model_dict = {
        "glow": flowlib.Glow,
    }
    model = model_dict[args_model](**config[f"{args_model}_params"])

    train_data, test_data = load_data(dataset_name, str(data_dir), config["image_size"])

    trainer = flowlib.Trainer(**params)
    trainer.run(model, train_data, test_data)


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML training")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA devices by comma separation.")
    parser.add_argument("--model", type=str, default="glow", help="Model name.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max-steps", type=int, default=2, help="Number of gradient steps.")
    parser.add_argument("--max-grad-value", type=float, default=5.0, help="Clipping value.")
    parser.add_argument("--max-grad-norm", type=float, default=100.0, help="Clipping norm.")
    parser.add_argument("--test-interval", type=int, default=2, help="Interval steps for testing.")
    parser.add_argument("--save-interval", type=int, default=2, help="Interval steps for saving.")
    parser.add_argument(
        "--y-conditional",
        action="store_true",
        help="Use labels for conditional model (defualt=False).",
    )

    return parser.parse_args()


def load_data(dataset_name: str, data_dir: str, image_size: int) -> Tuple[Dataset, Dataset]:
    if dataset_name == "cifar":
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
            "root": data_dir,
            "download": True,
            "train": True,
            "transform": trans_train,
        }
        test_kwargs = {
            "root": data_dir,
            "download": True,
            "train": False,
            "transform": trans_test,
        }

        train_data = datasets.CIFAR10(**train_kwargs)
        test_data = datasets.CIFAR10(**test_kwargs)

    elif dataset_name == "celeba":
        transform = transforms.Compose(
            [
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

        train_kwargs = {
            "root": data_dir,
            "download": True,
            "split": "train",
            "transform": transform,
        }
        test_kwargs = {
            "root": data_dir,
            "download": True,
            "split": "test",
            "transform": transform,
        }

        train_data = datasets.CelebA(**train_kwargs)
        test_data = datasets.CelebA(**test_kwargs)
    else:
        raise ValueError(f"Unexpected dataset name: {dataset_name}.")

    return train_data, test_data


if __name__ == "__main__":
    main()
