// =============================================================================
// hip_projection.hip — AMD GPU Projection Kernel (ROCm / HIP)
// Foveated 2.5D Grid Engine — PS 26053
// =============================================================================

#include "hip_projection.h"
#include <cmath>
#include <iostream>
#include <limits>

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#define HAS_HIP_RUNTIME 1
#else
#define HAS_HIP_RUNTIME 0
#endif

namespace foveated_grid {

#if HAS_HIP_RUNTIME

// Device-side atomic min for 32-bit floats via atomicCAS
__device__ inline void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

// Device-side atomic max for 32-bit floats via atomicCAS
__device__ inline void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

// Device-side atomic add for uint16_t fields
__device__ inline void atomicAddU16(uint16_t* address, uint16_t val) {
    uint32_t* address32 = (uint32_t*)((uintptr_t)address & ~3);
    bool is_high = ((uintptr_t)address & 2) != 0;
    uint32_t old = *address32, assumed;
    do {
        assumed = old;
        uint16_t cur = is_high ? (assumed >> 16) : (assumed & 0xFFFF);
        uint16_t next = cur + val;
        uint32_t updated = is_high ? ((assumed & 0x0000FFFF) | ((uint32_t)next << 16))
                                   : ((assumed & 0xFFFF0000) | (uint32_t)next);
        old = atomicCAS(address32, assumed, updated);
    } while (assumed != old);
}

// -----------------------------------------------------------------------------
// HIP GPU Kernel: Parallel Point Projection to Foveated Grid Cells
// -----------------------------------------------------------------------------
__global__ void projectPointsKernel(
    const ClassifiedPoint* points,
    int num_points,
    const ZoneConfigGPU* zones,
    int num_zones,
    CellGPU* grid_cells,
    int total_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    ClassifiedPoint p = points[idx];
    float r = sqrtf(p.x * p.x + p.y * p.y);

    // Find zone
    for (int z = 0; z < num_zones; ++z) {
        ZoneConfigGPU zone = zones[z];
        bool in_zone = (r >= zone.r_min && r < zone.r_max);
        if (!in_zone && z == num_zones - 1 && fabsf(r - zone.r_max) < 1e-4f) {
            in_zone = true;
        }
        if (in_zone) {
            int gx = (int)floorf(p.x / zone.cell_size) + zone.offset_x;
            int gy = (int)floorf(p.y / zone.cell_size) + zone.offset_y;
            if (gx == zone.grid_width && fabsf(p.x - zone.r_max) < 1e-4f) gx = zone.grid_width - 1;
            if (gy == zone.grid_height && fabsf(p.y - zone.r_max) < 1e-4f) gy = zone.grid_height - 1;

            if (gx >= 0 && gx < zone.grid_width && gy >= 0 && gy < zone.grid_height) {
                int cell_idx = zone.base_idx + gy * zone.grid_width + gx;
                if (cell_idx >= 0 && cell_idx < total_cells) {
                    CellGPU* cell = &grid_cells[cell_idx];

                    atomicMinFloat(&cell->elevation_min, p.z);
                    atomicMaxFloat(&cell->elevation_max, p.z);
                    atomicAdd(&cell->elevation_sum, p.z);

                    if (p.semantic_class < 6) {
                        atomicAdd(&cell->semantic_votes[p.semantic_class], 1);
                    }

                    atomicAdd(&cell->confidence_sum, p.confidence);
                    atomicAddU16(&cell->point_count, 1);
                }
            }
            break;
        }
    }
}

#endif // HAS_HIP_RUNTIME

bool isHipAvailable() {
#if HAS_HIP_RUNTIME
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    return (err == hipSuccess && device_count > 0);
#else
    return false;
#endif
}

bool launchHipProjection(
    const std::vector<ClassifiedPoint>& points,
    const std::vector<ZoneConfigGPU>& zones,
    std::vector<CellGPU>& out_cells
) {
    if (zones.empty()) return false;
    const auto& last_zone = zones.back();
    int total_cells = last_zone.base_idx + last_zone.total_cells;

    out_cells.resize(total_cells);

    // Initialize output cells with sentinels
    for (int i = 0; i < total_cells; ++i) {
        out_cells[i].elevation_min = 1e9f;
        out_cells[i].elevation_max = -1e9f;
        out_cells[i].elevation_sum = 0.0f;
        for (int c = 0; c < 6; ++c) {
            out_cells[i].semantic_votes[c] = 0;
        }
        out_cells[i].confidence_sum = 0.0f;
        out_cells[i].point_count = 0;
        out_cells[i]._padding = 0;
    }

    if (points.empty()) return true;

#if HAS_HIP_RUNTIME
    if (!isHipAvailable()) {
        std::cerr << "[HIP Projection] No active HIP GPU detected. Falling back to host." << std::endl;
        goto host_fallback;
    }

    ClassifiedPoint* d_points = nullptr;
    ZoneConfigGPU* d_zones = nullptr;
    CellGPU* d_cells = nullptr;

    hipError_t err;
    err = hipMalloc(&d_points, points.size() * sizeof(ClassifiedPoint));
    if (err != hipSuccess) goto host_fallback;

    err = hipMalloc(&d_zones, zones.size() * sizeof(ZoneConfigGPU));
    if (err != hipSuccess) { hipFree(d_points); goto host_fallback; }

    err = hipMalloc(&d_cells, total_cells * sizeof(CellGPU));
    if (err != hipSuccess) { hipFree(d_points); hipFree(d_zones); goto host_fallback; }

    hipMemcpy(d_points, points.data(), points.size() * sizeof(ClassifiedPoint), hipMemcpyHostToDevice);
    hipMemcpy(d_zones, zones.data(), zones.size() * sizeof(ZoneConfigGPU), hipMemcpyHostToDevice);
    hipMemcpy(d_cells, out_cells.data(), total_cells * sizeof(CellGPU), hipMemcpyHostToDevice);

    {
        int block_size = 256;
        int num_blocks = (static_cast<int>(points.size()) + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            projectPointsKernel,
            dim3(num_blocks),
            dim3(block_size),
            0,
            0,
            d_points,
            static_cast<int>(points.size()),
            d_zones,
            static_cast<int>(zones.size()),
            d_cells,
            total_cells
        );

        hipDeviceSynchronize();
    }

    hipMemcpy(out_cells.data(), d_cells, total_cells * sizeof(CellGPU), hipMemcpyDeviceToHost);

    hipFree(d_points);
    hipFree(d_zones);
    hipFree(d_cells);
    return true;

host_fallback:
#endif

    // Host emulation path (runs when HIP is unavailable or outside GPU environment)
    for (const auto& p : points) {
        float r = std::sqrt(p.x * p.x + p.y * p.y);
        for (size_t z = 0; z < zones.size(); ++z) {
            const auto& zone = zones[z];
            bool in_zone = (r >= zone.r_min && r < zone.r_max);
            if (!in_zone && z == zones.size() - 1 && std::abs(r - zone.r_max) < 1e-4f) {
                in_zone = true;
            }
            if (in_zone) {
                int gx = static_cast<int>(std::floor(p.x / zone.cell_size)) + zone.offset_x;
                int gy = static_cast<int>(std::floor(p.y / zone.cell_size)) + zone.offset_y;
                if (gx == zone.grid_width && std::abs(p.x - zone.r_max) < 1e-4f) gx = zone.grid_width - 1;
                if (gy == zone.grid_height && std::abs(p.y - zone.r_max) < 1e-4f) gy = zone.grid_height - 1;

                if (gx >= 0 && gx < zone.grid_width && gy >= 0 && gy < zone.grid_height) {
                    int cell_idx = zone.base_idx + gy * zone.grid_width + gx;
                    if (cell_idx >= 0 && cell_idx < total_cells) {
                        CellGPU& cell = out_cells[cell_idx];
                        if (p.z < cell.elevation_min) cell.elevation_min = p.z;
                        if (p.z > cell.elevation_max) cell.elevation_max = p.z;
                        cell.elevation_sum += p.z;
                        if (p.semantic_class < 6) {
                            cell.semantic_votes[p.semantic_class] += 1;
                        }
                        cell.confidence_sum += p.confidence;
                        cell.point_count += 1;
                    }
                }
                break;
            }
        }
    }

    return true;
}

} // namespace foveated_grid
