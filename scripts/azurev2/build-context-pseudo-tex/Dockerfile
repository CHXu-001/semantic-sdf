# Define base image.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:20230907.v1


ARG TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ARG CUDA_ARCHITECTURES=75


# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

ENV FORCE_CUDA="1"

ENV MAX_GCC_VERSION 11 

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc-$MAX_GCC_VERSION \
    g++-$MAX_GCC_VERSION \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*


RUN ln -s /usr/bin/gcc-$MAX_GCC_VERSION /usr/local/cuda/bin/gcc && \
    ln -s /usr/bin/g++-$MAX_GCC_VERSION /usr/local/cuda/bin/g++

# Create non root user and setup environment.
RUN useradd -m -d /home/user -g root -G sudo -u 1000 user
RUN usermod -aG sudo user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new user and workdir.
USER 1000
WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

# Upgrade pip and install packages.
RUN python3.10 -m pip install --upgrade pip setuptools pathtools promise pybind11 debugpy

# Install pytorch and submodules (Currently, we still use cu116 which is the latest version for torch 1.12.1 and is compatible with CUDA 11.8).
RUN python3.10 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118


RUN python3.10 -m pip install mlflow azureml-mlflow opencv-python tyro

RUN python3.10 -m pip install git+https://github.com/facebookresearch/detectron2  && \
    python3.10 -m pip install git+https://github.com/cocodataset/panopticapi.git  

RUN git clone https://github.com/facebookresearch/Mask2Former.git && \
    cd Mask2Former && \
    python3.10 -m pip install -r requirements.txt && \
    cd mask2former/modeling/pixel_decoder/ops && \
    python3.10 setup.py build install --user 


# Change working directory
WORKDIR /workspace
