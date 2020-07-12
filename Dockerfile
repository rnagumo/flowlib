FROM nvidia/cuda:11.0-runtime-ubuntu18.04-rc

# Install Python and utilities
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.7 python3.7-dev python3.7-venv python3-pip python3-wheel \
        python3-setuptools \
        git vim ssh wget gcc cmake build-essential libblas3 libblas-dev \
    && rm /usr/bin/python3 \
    && ln -s python3.7 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy package
WORKDIR /app
COPY bin/ bin/
COPY examples/ examples/
COPY flowlib/ flowlib/
COPY tests/ tests/
COPY setup.py setup.py

# Install package
RUN pip install --upgrade pip
RUN pip install -e .

# Install other requirements for examples
RUN pip install --no-cache-dir numpy==1.19.0 pandas==1.0.5 matplotlib==3.2.2 \
        torchvision==0.6.1 tqdm==4.46.1  tensorboardX==2.0
