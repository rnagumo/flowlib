
"""Plot example."""

import argparse

import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torchvision import transforms, datasets
from torchvision.utils import make_grid

import flowlib


def main() -> None:
    """Load pre-trained model and show reconstruction and samples."""

    # Settings
    parser = argparse.ArgumentParser(description="Flow training")
    parser.add_argument("--root", type=str, default="./data/cifar/",
                        help="Path to root directory of data.")
    parser.add_argument("--cp-path", type=str, default="./logs/tmp/cp.pt",
                        help="Path to checkpoint file.")
    args = parser.parse_args()

    # Load pre-trained model
    model = flowlib.Glow()
    cp = torch.load(args.cp_path, map_location=torch.device("cpu"))
    model.load_state_dict(cp["model_state_dict"])

    # Data
    trans_test = transforms.Compose([transforms.ToTensor()])
    test_kwargs = {"root": args.root, "train": False, "download": False,
                   "transform": trans_test}
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(**test_kwargs), shuffle=False, batch_size=16)
    data, _ = next(iter(loader))

    # Reconstruct and sample
    with torch.no_grad():
        z = model(data)
        recon = model.inverse(z)

        sample = model.sample(16)

    # Show grid
    plt.figure(figsize=(16, 12))

    plt.subplot(311)
    gridshow(data)
    plt.title("Original")

    plt.subplot(312)
    gridshow(recon)
    plt.title("Reconstructed")

    plt.subplot(313)
    gridshow(sample)
    plt.title("Sampled")

    plt.tight_layout()
    plt.show()


def imshow(img: Tensor) -> None:
    """Show single image.

    Args:
        img (torch.Tensor): (c, h, w) or (1, c, h, w).
    """

    if img.dim() == 4 and img.size(0) == 1:
        img = img.squeeze(0)
    elif img.dim() != 3:
        raise ValueError(f"Wrong image size: {img.size()}")

    # CHW -> HWC
    npimg = img.permute(1, 2, 0).numpy()
    plt.imshow(npimg, interpolation="nearest")


def gridshow(img: Tensor) -> None:
    """Show images in grid.

    Args:
        img (torch.Tensor): (b, c, h, w) or (b, 1, c, h, w).
    """

    if img.dim() == 5 and img.size(1) == 1:
        img = img.squeeze(1)
    elif img.dim() != 4:
        raise ValueError(f"Wrong image size: {img.size()}")

    grid = make_grid(img)
    npgrid = grid.permute(1, 2, 0).numpy()
    plt.imshow(npgrid, interpolation="nearest")


if __name__ == "__main__":
    main()
