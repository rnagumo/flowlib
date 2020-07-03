
from setuptools import setup, find_packages


install_requires = [
    "torch==1.5.1",
    "numpy==1.19.0",
    "pandas==1.0.5",
]


setup(
    name="flowlib",
    version="0.1",
    description="Flow models by PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
)
