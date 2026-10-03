# AMD ROCm Setup & Verification Guide

This guide details the hardware and software configuration required for running the **Adaptive Variable Resolution 2.5D Lidar Mapping** pipeline on AMD Radeon and Instinct GPUs.

---

## 1. Supported Hardware & Requirements

- **Supported GPUs**:
  - AMD Radeon RX 7900 XTX / 7900 XT / 7900 GRE (RDNA 3 / `gfx1100`)
  - AMD Radeon RX 7800 XT / 7700 XT (`gfx1101`)
  - AMD Radeon RX 6900 XT / 6800 XT (RDNA 2 / `gfx1030` — with override)
  - AMD Instinct MI200 / MI300 series (CDNA)
- **Minimum VRAM**: 16 GB (Recommended: 24 GB)
- **System RAM**: ≥ 32 GB
- **Operating System**: Ubuntu 22.04 LTS or Ubuntu 24.04 LTS (x86_64)
- **ROCm Version**: 6.0, 6.1, or 6.2+

---

## 2. Host Verification

Verify that the AMD kernel driver (`amdgpu`) and ROCm stack are active:

```bash
# 1. Verify GPU is detected by ROCm
rocminfo | grep "Name"

# 2. Check GPU telemetry and VRAM status
rocm-smi

# 3. Check HIP compiler version
hipcc --version
```

### Architecture Override (RDNA 2 / RX 6000 series)
If running on RX 6000 series (such as RX 6800 / 6900 XT), export the HSA GFX override before launching Python or PyTorch:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

---

## 3. Docker Environment (Recommended)

Using Docker with ROCm passthrough ensures binary compatibility without dependency conflicts.

### Build and Launch Docker Container

```bash
# 1. Build the training container
docker build -t lidar25d-train -f docker/Dockerfile.train .

# 2. Run with GPU passthrough
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --shm-size=16g \
  -v $(pwd):/workspace \
  lidar25d-train bash
```

### In-Container Verification

```python
import torch
print("ROCm / CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
print("Device Name:", torch.cuda.get_device_name(0))
```

---

## 4. Sparse Convolutions with spconv-triton

Traditional CUDA libraries (such as MinkowskiEngine) are NVIDIA-specific and will not compile on ROCm. This project utilizes `spconv-triton`, an AMD-compatible sparse convolution backend utilizing OpenAI Triton.

Verify sparse convolution functionality:

```python
import torch
import spconv.pytorch as spconv

# Verify sparse tensor allocation on AMD GPU
features = torch.randn(100, 16).cuda()
indices = torch.randint(0, 128, (100, 4)).int().cuda()
spatial_shape = [128, 128, 128]

sp_tensor = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size=1)
print(f"Sparse tensor initialized on: {sp_tensor.features.device}")
```

---

## 5. Building C++ / HIP Grid Engine

The foveated grid engine is built with CMake and the HIP compiler:

```bash
bash scripts/build_grid_engine.sh
```

This compiles:
- `libgrid_engine.a`: C++ static library.
- `libgrid_hip_kernels.a`: AMD GPU HIP kernels.
- `grid_py.cpython-*.so`: pybind11 Python extension.
- `test_grid_engine`: Google Test C++ test runner.

---

## 6. SemanticKITTI Dataset Setup

To train on SemanticKITTI:

1. Download KITTI Odometry point clouds (`data_odometry_velodyne.zip`) from [KITTI Vision Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
2. Download SemanticKITTI label files (`data_odometry_labels.zip`) from [SemanticKITTI](http://www.semantic-kitti.org/).
3. Unzip into `data/semantickitti/dataset/sequences/`:
   ```
   data/semantickitti/dataset/sequences/
   ├── 00/
   │   ├── velodyne/
   │   │   ├── 000000.bin
   │   │   └── ...
   │   └── labels/
   │       ├── 000000.label
   │       └── ...
   ├── 01/
   └── ...
   ```
4. Update dataset path in [config/model_config.yaml](../config/model_config.yaml) if located elsewhere.
