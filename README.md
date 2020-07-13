
# flowlib

Flow models in PyTroch.

# Requirements

* Python == 3.7
* PyTorch == 1.5.1

Additional requirements for example codes.

* numpy == 1.19.0
* pandas == 1.0.5
* matplotlib == 3.2.2
* torchvision == 0.6.1
* tqdm == 4.46.1
* tensorboardX == 2.0

# Setup

Clone repository.

```bash
git clone https://github.com/rnagumo/flowlib.git
cd flowlib
```

Install the package in virtual env.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install .
```

Or use [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker). You can run container with GPUs by Docker 19.03+.

```bash
docker build -t flowlib .
docker run --gpus all -it flowlib bash
```

Install other requirements for example code.

```bash
pip3 install numpy==1.19.0 pandas==1.0.5 matplotlib==3.2.2 torchvision==0.6.1 tqdm==4.46.1  tensorboardX==2.0
```

# Experiment

Run the shell script in `bin` directory. See the script for the experimental detail.

* random-seed: Random seed for reproduction.
* conditional-flag: Boolean flag for conditional model that uses labels for training.

```bash
# Usage
bash bin/train.sh <random-seed> <conditional-flag>

# Example (non conditional model)
bash bin/train.sh 123

# Example (conditional model)
bash bin/train.sh 123 1
```

# Example code

## Training

```python
import torch
from torch import optim
from torchvision import transforms, datasets

import flowlib


# Dataset
loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data/cifar", train=True,
                     transform=transforms.ToTensor(), download=True),
    shuffle=True, batch_size=32,
)

# Model
model = flowlib.Glow()
optimizer = optim.Adam(model.parameters())

for data, _ in loader:
    model.train()
    optimizer.zero_grad()

    loss_dict = model.loss_func(data)
    loss = loss_dict["loss"].mean()
    loss.backward()
    optimizer.step()
```

## Qualitative Evaluation

```python
import torch
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import flowlib


# Dataset
loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data/cifar", train=False,
                     transform=transforms.ToTensor(), download=True),
    shuffle=False, batch_size=16,
)

# Model
model = flowlib.Glow()

# Reconstruct and sample
model.eval()
data, _ = next(iter(loader))
with torch.no_grad():
    recon = model.reconstruct(data)
    sample = model.sample(16)


def gridshow(img):
    grid = make_grid(img)
    npgrid = grid.permute(1, 2, 0).numpy()
    plt.imshow(npgrid, interpolation="nearest")


plt.figure(figsize=(20, 12))

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
```

# Reference

## Original papers

* D. Rezende *et al*., "Variational Inference with Normalizing Flows." (ICML 2015) [arXiv](http://arxiv.org/abs/1505.05770)
* D. Kingma *et al*., "Improved Variational Inference with Inverse Autoregressive Flow." (NeurIPS 2016) [arXiv](http://arxiv.org/abs/1606.04934)
* L. Dinh *et al*., "NICE: Non-linear independent components estimation." (ICLR 2015) [arXiv](https://arxiv.org/abs/1410.8516v6)
* M. Germain *et al*., "MADE: Masked Autoencoder for Distribution Estimation." (ICML 2015) [arXiv](http://arxiv.org/abs/1502.03509)
* G. Papamakarios *et al*., "Masked autoregressive flow for density estimation." (NeurIPS 2017) [arXiv](http://arxiv.org/abs/1705.07057)
* L. Dinh *et al*., "Density estimation using Real NVP." (ICLR 2017) [arXiv](http://arxiv.org/abs/1605.08803)
* D. Kingma *et al*., "Glow: Generative Flow with Invertible 1x1 Convolutions." (NeurIPS 2018) [arXiv](http://arxiv.org/abs/1807.03039)

## Codes

* OpenAI, glow (Glow implementation in TensorFlow by Authors) [GitHub](https://github.com/openai/glow)
* ikostrikov, pytorch-flows [GitHub](https://github.com/ikostrikov/pytorch-flows)
* y0ast, Glow-PyTorch [GitHub](https://github.com/y0ast/Glow-PyTorch)
* ex4sperans, variational-inference-with-normalizing-flows [GitHub](https://github.com/ex4sperans/variational-inference-with-normalizing-flows)
* rosinality, glow-pytorch [GitHub](https://github.com/rosinality/glow-pytorch)
* chaiyujin, glow-pytorch [GitHub](https://github.com/chaiyujin/glow-pytorch)
* gpapamak, maf [GitHub](https://github.com/gpapamak/maf)
* masa-su, pixyz, Glow (CIFAR 10) [GitHub](https://github.com/masa-su/pixyz/blob/master/examples/glow.ipynb), Real NVP (CIFAR10) [GitHub](https://github.com/masa-su/pixyz/blob/master/examples/real_nvp_cifar.ipynb)
* pclucas14, pytorch-glow [GitHub](https://github.com/pclucas14/pytorch-glow)
* taesungp, real-nvp [GitHub](https://github.com/taesungp/real-nvp)
* tensorflow, Real NVP in TensorFlow [GitHub](https://github.com/tensorflow/models/tree/master/research/real_nvp)
