FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 \
        pybind11-dev \
        libeigen3-dev

RUN pip3 install torch==1.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"


# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--blas=openblas"


COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /packages

# legacy packages needed for legacy data loaders based on Smart-Tree
RUN git clone --recursive https://github.com/lxxue/FRNN.git
RUN pip install FRNN/external/prefix_sum/. 
RUN pip install -e FRNN/.
RUN git clone https://github.com/PRBonn/DiffTS.git

WORKDIR /packages/DiffTS
RUN pip3 install -U -e .
WORKDIR /packages

RUN pip install cugraph-cu11 --extra-index-url=https://pypi.nvidia.com


ARG USER_ID
ARG GROUP_ID

# Switch to same user as host system
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /packages/DiffTS/DiffTS