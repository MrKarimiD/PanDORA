#!/usr/bin/env bash
#
# Build the PanDORA docker image and run the fixed two-stage training pipeline
# inside it, with GPU access and the repo + dataset mounted at runtime.
#
# Usage:
#   docker/build_and_run.sh [--rebuild] [--shell]
#
#   --rebuild   Rebuild the image from scratch (docker build --no-cache).
#   --shell     Drop into an interactive shell instead of running the pipeline.
#
# Common environment overrides:
#   DATA_ROOT     Host dir mounted to /data      (default below)
#   DATASET_DIR   Container path to a *_ns scene (default below, under /data)
#   STEP1_ITERS / STEP2_ITERS   Lower for a quick smoke test.
set -euo pipefail

# --- Configuration -----------------------------------------------------------
IMAGE="${IMAGE:-pandora:latest}"
# Repo root = parent of this script's directory.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Host data directory that holds the processed scenes; mounted to /data.
DATA_ROOT="${DATA_ROOT:-/home-local2/${USER}.extra.nobkp/nerfstudio_ds}"
# Container-side path to the scene to train on (lives under /data).
DATASET_DIR="${DATASET_DIR:-/data/sample_scene/sample_scene_ns/}"
SHM_SIZE="${SHM_SIZE:-12gb}"

REBUILD=0
RUN_SHELL=0
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        --shell)   RUN_SHELL=1 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# --- Build -------------------------------------------------------------------
# Parallel compile jobs for the from-source builds (ceres/colmap/tcnn). Keep
# modest to avoid GCC OOM/segfault on the heavy Ceres template files; raise on
# high-RAM machines, e.g. MAKE_JOBS=8 docker/build_and_run.sh.
MAKE_JOBS="${MAKE_JOBS:-4}"
# CUDA toolkit version for the base image + torch/tcnn build. Must be <= the max
# CUDA version your host NVIDIA driver supports (see `nvidia-smi`). Default 11.7.1
# (torch 2.0.1+cu117) matches the known-working env and runs on older drivers
# (r450+). torch 2.0.1 also ships cu118 wheels if your driver is newer.
CUDA_VERSION="${CUDA_VERSION:-11.7.1}"
BUILD_ARGS=(-t "$IMAGE" -f "$REPO/Dockerfile"
    --build-arg MAKE_JOBS="$MAKE_JOBS"
    --build-arg CUDA_VERSION="$CUDA_VERSION"
    "$REPO")
if [ "$REBUILD" -eq 1 ]; then
    echo ">> Rebuilding image '$IMAGE' from scratch (--no-cache)..."
    docker build --no-cache "${BUILD_ARGS[@]}"
else
    echo ">> Building image '$IMAGE' (cached)..."
    docker build "${BUILD_ARGS[@]}"
fi

# --- Run ---------------------------------------------------------------------
# Mount the repo onto the same path the image installed it editable (-e .), so
# the live code is used and the editable install stays valid.
DOCKER_RUN=(
    docker run --rm -it
    --gpus all
    --shm-size="$SHM_SIZE"
    -v "$REPO":/home/user/nerfstudio
    -v "$DATA_ROOT":/data
    -e DATASET_DIR="$DATASET_DIR"
    -e EXPERIMENT_NAME="${EXPERIMENT_NAME:-pandora}"
    -e STEP1_ITERS="${STEP1_ITERS:-60000}"
    -e STEP2_ITERS="${STEP2_ITERS:-120000}"
    -w /home/user/nerfstudio
    "$IMAGE"
)

if [ "$RUN_SHELL" -eq 1 ]; then
    echo ">> Launching interactive shell in '$IMAGE'..."
    exec "${DOCKER_RUN[@]}" bash
else
    echo ">> Running PanDORA pipeline in '$IMAGE' on DATASET_DIR=$DATASET_DIR ..."
    exec "${DOCKER_RUN[@]}" bash docker/run_pandora_pipeline.sh
fi
