#pragma once
// =============================================================================
// cell.h — Grid Cell Data Structure
// Foveated 2.5D Grid Engine — PS 26053
// =============================================================================

#include <cstdint>
#include <limits>
#include <atomic>

namespace foveated_grid {

/// Represents a single cell in the 2.5D elevation grid.
/// Each cell stores elevation statistics, semantic class, and occupancy state.
struct Cell {
    float    elevation_min  = std::numeric_limits<float>::max();
    float    elevation_max  = std::numeric_limits<float>::lowest();
    float    elevation_sum  = 0.0f;   // For computing mean
    uint8_t  semantic_class = 5;      // Default: unknown/noise
    float    confidence     = 0.0f;   // Segmentation confidence [0, 1]
    uint8_t  occupancy      = 2;      // 0=free, 1=occupied, 2=unknown
    uint16_t point_count    = 0;      // Number of points projected into this cell

    /// Reset cell to default state (call at start of each frame)
    void reset() {
        elevation_min  = std::numeric_limits<float>::max();
        elevation_max  = std::numeric_limits<float>::lowest();
        elevation_sum  = 0.0f;
        semantic_class = 5;
        confidence     = 0.0f;
        occupancy      = 2;
        point_count    = 0;
    }

    /// Compute mean elevation (call after all points are projected)
    float elevation_mean() const {
        return (point_count > 0) ? (elevation_sum / point_count) : 0.0f;
    }

    /// Height span of the cell (max - min elevation)
    float height_span() const {
        return (point_count > 0) ? (elevation_max - elevation_min) : 0.0f;
    }
};

/// GPU-compatible version of Cell for HIP kernels (POD, no std:: members)
struct alignas(32) CellGPU {
    float    elevation_min;    // Use __float_as_int for atomic ops
    float    elevation_max;
    float    elevation_sum;
    int      semantic_votes[6]; // Vote count per class (atomicAdd)
    float    confidence_sum;
    uint16_t point_count;
    uint16_t _padding;         // Alignment padding
};

} // namespace foveated_grid
