from setuptools import setup, find_packages


install_requires = [
    "torch>=1.6",
    "torchvision>=0.7",
]


extras_require = {
    "training": [
        "numpy>=1.19",
        "pandas>=1.0",
        "matplotlib>=3.2",
        "tqdm>=4.48",
        "tensorboardX>=2.1",
    ],
    "dev": [
        "pytest",
        "black",
        "flake8",
        "mypy==0.790",
    ],
}


setup(
    name="flowlib",
    version="0.2",
    description="Flow models in PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
