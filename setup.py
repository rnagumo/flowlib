
from setuptools import setup, find_packages


install_requires = [
    "torch==1.5.1",
    "torchvision==0.6.1",
    "tqdm==4.46.1",
    "tensorboardX==2.0",
    "matplotlib==3.2.2",
]


setup(
    name="flowlib",
    version="0.1",
    description="Flow models by PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
)
