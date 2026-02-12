# Dockerfile for OpenPI training on CHTC GPU nodes.
# Based on the existing serve_policy.Dockerfile, but installs the full
# project (including training dependencies) and uses CUDA 12.2 devel
# for JAX GPU compilation support.
#
# Build:
#   docker build -t openpi_train -f chtc/train.Dockerfile .
#
# Push to DockerHub (required for CHTC):
#   docker tag openpi_train <dockerhub_user>/openpi_train:latest
#   docker push <dockerhub_user>/openpi_train:latest

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs linux-headers-generic build-essential clang curl \
    ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy from cache instead of linking (container filesystem)
ENV UV_LINK_MODE=copy
# venv outside project so mounted volumes don't clobber it
ENV UV_PROJECT_ENVIRONMENT=/.venv
# Install Python to a world-readable location (default /root/.local is 700)
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

# Install Python and project dependencies from the lock file
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Copy the full project into the image
COPY . /app

# Install the project itself
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Bundle the PaliGemma tokenizer (GCS bucket is no longer publicly accessible).
# The file was cached locally from a previous download.
COPY chtc/assets/paligemma_tokenizer.model /opt/openpi-cache/big_vision/paligemma_tokenizer.model

# CHTC runs containers as a non-root user; make everything readable.
RUN chmod -R a+rX $UV_PROJECT_ENVIRONMENT /opt/openpi-cache

ENV PATH="/.venv/bin:$PATH"

CMD ["/bin/bash"]
