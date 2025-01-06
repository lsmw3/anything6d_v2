# Start from NVIDIA CUDA 11.8 base image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Avoid timezone prompt during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    ninja-build \
    bzip2 \
    ca-certificates \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# # Set the environment variables for CUDA 11.8
# ENV CUDA_HOME=/usr/local/cuda-11.8
# ENV PATH=/usr/local/cuda-11.8/bin:$PATH
# ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# # check if there's nvcc
# RUN nvcc --version

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=$CONDA_DIR/bin:$PATH

# Set working directory to lara project
WORKDIR /anything6d

# Copy the entire project directory
COPY . .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "$(grep name environment.yml | cut -d' ' -f2)", "/bin/bash", "-c"]

# Install xformers with pip
RUN pip3 install -U xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

# Set the default command to run the training script
CMD ["conda", "run", "-n", "$(grep name environment.yml | cut -d' ' -f2)", "python", "train_lightning.py"]