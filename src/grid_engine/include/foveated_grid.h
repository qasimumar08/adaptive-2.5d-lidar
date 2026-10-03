#pragma once
// =============================================================================
// foveated_grid.h — Top-Level Foveated 2.5D Grid API
// Foveated 2.5D Grid Engine — PS 26053
// =============================================================================

#include "cell.h"
#include "zone_manager.h"
#include <vector>
#include <array>
#include <cstdint>

namespace foveated_grid {

/// Classified 3D point — output from semantic segmentation.
struct ClassifiedPoint {
    float x, y, z;
    uint8_t semantic_class;  // 0-5
    float   confidence;      // [0, 1]
};

/// Main API for the foveated 2.5D grid.
class FoveatedGrid {
public:
    /// Construct with a configured ZoneManager.
    explicit FoveatedGrid(const ZoneManager& zone_manager);

    /// Reset all cells to default state (call at start of each frame).
    void clear();

    /// Project classified points into the grid (CPU path).
    /// Points are binned into zones by radial distance and accumulated into cells.
    void projectPoints(const std::vector<ClassifiedPoint>& points);

    /// Project classified points using the HIP GPU kernel (AMD GPU path).
    /// Uploads points to GPU, runs kernel, downloads results.
    void projectPointsGPU(const std::vector<ClassifiedPoint>& points);

    /// Finalize cells after projection: compute mean elevation, majority vote
    /// for semantic class, set occupancy based on point_count threshold.
    void finalizeCells(int unknown_threshold = 0, float confidence_threshold = 0.5f);

    /// Access a cell by zone index and 2D grid position.
    const Cell& getCell(int zone_idx, int grid_x, int grid_y) const;

    /// Access the full cell array (flat, ordered by zone).
    const std::vector<Cell>& cells() const { return cells_; }

    /// Get total number of cells.
    int totalCells() const { return static_cast<int>(cells_.size()); }

    /// Get memory usage in bytes.
    size_t memoryUsage() const { return cells_.size() * sizeof(Cell); }

    /// Get the zone manager.
    const ZoneManager& zoneManager() const { return zone_manager_; }

private:

    ZoneManager zone_manager_;
    std::vector<Cell> cells_;
    std::vector<std::array<int, 6>> class_votes_;
    std::vector<float> conf_sums_;
    std::vector<int> modified_indices_;
};

} // namespace foveated_grid

