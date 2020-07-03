
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

Install other libraries for example code.

```bash
pip3 install numpy==1.19.0 pandas==1.0.5 matplotlib==3.2.2 torchvision==0.6.1 tqdm==4.46.1  tensorboardX==2.0
```

Or use [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

```bash
docker build -t flowlib .
docker run -it flowlib bash
```

You can run container with GPUs by Docker 19.03.

```bash
docker run --gpus all -it flowlib bash
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
