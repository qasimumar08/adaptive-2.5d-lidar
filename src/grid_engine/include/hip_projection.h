#pragma once
// =============================================================================
// hip_projection.h — Host Interface for HIP GPU Point Projection Kernel
// Foveated 2.5D Grid Engine — PS 26053
// =============================================================================

#include "cell.h"
#include "foveated_grid.h"
#include "zone_manager.h"
#include <vector>

namespace foveated_grid {

/// Launch parallel HIP kernel on AMD GPU to project points into foveated grid cells.
/// Returns true if GPU execution succeeded, false if HIP failed or was unavailable.
bool launchHipProjection(
    const std::vector<ClassifiedPoint>& points,
    const std::vector<ZoneConfigGPU>& zones,
    std::vector<CellGPU>& out_cells
);

/// Check whether HIP runtime and an AMD GPU device are currently active and available.
bool isHipAvailable();

} // namespace foveated_grid
