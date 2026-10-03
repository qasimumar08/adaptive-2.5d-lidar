#!/usr/bin/env bash
# =============================================================================
# build_grid_engine.sh — Build Foveated Grid Engine C++ & Python Bindings
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Building Foveated Grid Engine (C++ / HIP / pybind11) ==="
cd "${ROOT_DIR}"

mkdir -p build
cd build
cmake ../src/grid_engine "$@"
make -j"$(nproc)"

echo "=== Build Complete! Artifacts in build/ ==="
ls -lh grid_py*.so libgrid_engine.a libgrid_hip_kernels.a test_grid_engine 2>/dev/null || true
