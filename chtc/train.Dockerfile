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
    && rm -rf /var/lib/apt/lists/*

# Copy from cache instead of linking (container filesystem)
ENV UV_LINK_MODE=copy
# venv outside project so mounted volumes don't clobber it
ENV UV_PROJECT_ENVIRONMENT=/.venv

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

# CHTC runs containers as a non-root user; make the venv readable.
RUN chmod -R a+rX $UV_PROJECT_ENVIRONMENT

CMD ["/bin/bash"]
