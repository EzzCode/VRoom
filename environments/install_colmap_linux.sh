#!/bin/bash
set -e

echo "=========================================================="
echo "    Compiling COLMAP 3.10 from source (CUDA Enabled)      "
echo "=========================================================="

echo "[1/4] Installing dependencies via apt..."
sudo apt-get update
sudo apt-get install -y \
    cmake \
    ninja-build \
    build-essential \
    libmetis-dev \
    libfreeimage-dev \
    liblz4-dev \
    libceres-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libboost-all-dev \
    libsuitesparse-dev \
    git

echo "[2/4] Cloning COLMAP 3.10..."
rm -rf colmap-source
git clone --branch 3.10 --depth 1 https://github.com/colmap/colmap.git colmap-source

echo "[3/4] Configuring CMake..."
cd colmap-source
mkdir -p build
cd build

# Modify CMAKE_CUDA_ARCHITECTURES based on your GPU if necessary (e.g., 75 for Turing, 80 for Ampere, 86 for Ada/Lovelace)
# We set GUI_ENABLED=OFF since this is typically for headless backend processing or WSL.
cmake .. -GNinja \
    -DCUDA_ENABLED=ON \
    -DGUI_ENABLED=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="native"

echo "[4/4] Building and Installing..."
ninja
sudo ninja install

echo "=========================================================="
echo "✅ COLMAP installed successfully!"
echo "You can verify by running: colmap help"
echo "=========================================================="
