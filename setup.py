
from setuptools import setup, find_packages


install_requires = [
    "torch>=1.6",
]


extras_require = {
    "dev": [
        "pytest",
        "black",
        "flake8",
        "mypy",
    ],
    "example": [
        "torchvision>=0.7",
        "numpy>=1.19",
        "pandas>=1.0",
        "matplotlib>=3.2",
        "tqdm>=4.48",
        "tensorboardX>=2.1",
    ],
}


setup(
    name="flowlib",
    version="0.1",
    description="Flow models in PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
