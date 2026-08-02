# Default to CUDA 11.7 (torch 2.0.1+cu117) to match the known-working environment.
# cu117 runs on older NVIDIA drivers (r450+) via CUDA-11 minor-version compat;
# cu118 requires a newer driver. Override with --build-arg CUDA_VERSION=... .
ARG CUDA_VERSION=11.7.1
ARG OS_VERSION=22.04
ARG USER_ID=1000
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}

# metainformation
LABEL org.opencontainers.image.version = "0.1.18"
LABEL org.opencontainers.image.source = "https://github.com/nerfstudio-project/nerfstudio"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}"

# Variables used at build time.
## CUDA architectures for tiny-cuda-nn.
## NOTE: sm_89 (Ada) and sm_90 (Hopper) require CUDA >= 11.8, so they are omitted
## here to stay compatible with the default cu117 toolkit. To speed up the build,
## keep only your GPU's architecture (find it at https://developer.nvidia.com/cuda-gpus,
## e.g. 8.6 -> 86). Add 89;90 back only if you also bump CUDA_VERSION to >= 11.8.0.
ARG CUDA_ARCHITECTURES=86;80;75;70;61;52;37

# Cap parallel compile jobs for the from-source builds (glog/ceres/colmap/tcnn).
# Ceres+Eigen template compilation is memory-heavy: `make -j$(nproc)` on a
# many-core host can exhaust RAM and crash GCC (segfault ICE). Raise this only
# if the machine has plenty of RAM, e.g. --build-arg MAKE_JOBS=$(nproc).
ARG MAKE_JOBS=4

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
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


# NOTE: The COLMAP/Ceres/GLOG stack (and pycolmap/hloc/pyceres/pixel-perfect-sfm
# below) from the upstream Nerfstudio image has been removed. PanDORA obtains
# camera poses from OpenSFM (run outside this container) and trains on already
# processed *_ns scenes, so `ns-train PanDORA` never uses COLMAP-based SfM.
# Dropping it avoids a long, memory-heavy C++ build and shrinks the image.

# Create non root user and setup environment.
# Re-declare USER_ID inside the build stage: ARGs declared before FROM are only
# in scope for the FROM line, so ${USER_ID} would otherwise be empty in RUN steps.
# (Placed here, after the cached colmap build, to avoid invalidating that layer.)
ARG USER_ID=1000
RUN useradd -m -d /home/user -g root -G sudo -u ${USER_ID} user
RUN usermod -aG sudo user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new uer and workdir.
USER ${USER_ID}
WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

# Upgrade pip and install packages.
RUN python3.10 -m pip install --upgrade pip setuptools pathtools promise pybind11
# Install pytorch and submodules
# Re-declare CUDA_VERSION in the build stage (same before-FROM scope issue as USER_ID).
ARG CUDA_VERSION
RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python3.10 -m pip install \
    torch==2.0.1+cu${CUDA_VER} \
    torchvision==0.15.2+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

# CRITICAL: pin torch/torchvision/numpy for ALL later pip installs via a global
# constraints file + pip.conf. nerfstudio (torch>=1.13.1) and several extras have
# unbounded/newer torch requirements; without this they upgrade torch to the latest
# CUDA-13 build, which the host driver (CUDA 11.x) cannot run -> CUDA unavailable.
#  - pyequilib==0.5.6: newer pyequilib (>=0.6) dropped the Equi2Pers class that
#    nerfstudio's equirect_utils imports; 0.5.6 is the last version that keeps it.
RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && \
    printf 'torch==2.0.1+cu%s\ntorchvision==0.15.2+cu%s\nnumpy<2\npyequilib==0.5.6\n' "$CUDA_VER" "$CUDA_VER" > /home/user/pip-constraints.txt && \
    mkdir -p /home/user/.config/pip && \
    printf '[global]\nextra-index-url = https://download.pytorch.org/whl/cu%s\n' "$CUDA_VER" > /home/user/.config/pip/pip.conf
ENV PIP_CONSTRAINT=/home/user/pip-constraints.txt

# Install tynyCUDNN (we need to set the target architectures as environment variable first).
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
# Cap ninja/nvcc parallelism (same OOM concern as the ceres build above).
ENV MAX_JOBS=${MAKE_JOBS}
# tiny-cuda-nn's setup.py imports torch at build time. Modern pip builds wheels in
# an isolated env that lacks torch, so use --no-build-isolation to see the torch
# installed above; ninja is required to compile the CUDA bindings.
# Note: tiny-cuda-nn has no `v1.7` git tag (that version was only ever on master);
# v1.6 is the last real tag, builds on CUDA 11.7, and matches nerfstudio 0.3.2.
RUN python3.10 -m pip install ninja wheel && \
    python3.10 -m pip install --no-build-isolation \
        git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch

# (pycolmap / hloc / pyceres / pixel-perfect-sfm removed — see note above. These
# are COLMAP-based SfM feature matchers that the PanDORA pipeline does not use.)

RUN python3.10 -m pip install omegaconf
# Copy nerfstudio folder and give ownership to user.
ADD . /home/user/nerfstudio
USER root
RUN chown -R user /home/user/nerfstudio
USER ${USER_ID}

# Install nerfstudio dependencies.
RUN cd nerfstudio && \
    python3.10 -m pip install -e . && \
    cd ..

# ---------------------------------------------------------------------------
# PanDORA additions (on top of the base Nerfstudio image)
# ---------------------------------------------------------------------------

# Blender (pinned LTS). hdr_blender.py / render_all.py invoke `blender --background`,
# so we need the Blender binary on PATH (not the pip `bpy` module).
USER root
ARG BLENDER_VERSION=3.6.5
ARG BLENDER_MAJOR=3.6
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libsm6 xz-utils unzip && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q https://download.blender.org/release/Blender${BLENDER_MAJOR}/blender-${BLENDER_VERSION}-linux-x64.tar.xz -O /tmp/blender.tar.xz && \
    mkdir -p /opt/blender && \
    tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 && \
    rm /tmp/blender.tar.xz && \
    ln -s /opt/blender/blender /usr/local/bin/blender

# PanDORA-specific Python dependencies (as the non-root user).
USER ${USER_ID}
RUN python3.10 -m pip install --no-cache-dir \
        scikit-surgerycore \
        pydub \
        skylibs \
        piq \
        OpenEXR \
        Imath

# equilib: provided by pyequilib==0.5.6, pinned in the constraints file above and
# installed with nerfstudio. We deliberately do NOT use the repo's vendored
# equilib/ folder: it is git-ignored (absent on a fresh clone) and, because the
# repo root is on PYTHONPATH, a folder literally named `equilib/` shadows the real
# package as an empty namespace ("Equi2Pers unknown location").

# NOTE: lang-segment-anything is intentionally NOT installed. Current lang-sam
# requires torch>=2.3 (needs a newer driver than this cu117 build targets) and is
# only used for optional mask generation — the released dataset scenes already
# ship masks. Install a torch-2.0-compatible lang-sam separately if you need it.

# Pin runtime dependency versions that the floating requirements otherwise break:
#  - numpy<2: opencv-python==4.6.0.66 (and the torch 2.0.1 stack) are compiled
#    against numpy 1.x; some extras above pull numpy 2.x, breaking cv2 import.
#  - viser==0.0.16: nerfstudio 0.3.2's viewer_beta uses the old viser GUI API
#    (GuiHandle, ...) that newer viser (>=0.1.0) removed.
# Kept last so earlier installs can't override these pins.
RUN python3.10 -m pip install --no-cache-dir "numpy<2" "viser==0.0.16"

# Make the repo root importable so `lantern_scripts` (used by nerfstudio's lantern
# processors and PanDORA config) resolves without a manual `export PYTHONPATH`.
# The host repo is mounted onto this same path at runtime.
ENV PYTHONPATH=/home/user/nerfstudio

# Change working directory
WORKDIR /workspace

# Install nerfstudio cli auto completion and enter shell if no command was provided.
CMD ns-install-cli --mode install && /bin/bash

