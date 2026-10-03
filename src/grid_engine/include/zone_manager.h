#pragma once
// =============================================================================
// zone_manager.h — Multi-Resolution Zone Configuration
// Foveated 2.5D Grid Engine — PS 26053
// =============================================================================

#include <cstdint>
#include <vector>
#include <string>

namespace foveated_grid {

/// Configuration for a single resolution zone in the foveated grid.
struct ZoneConfig {
    std::string name;       // e.g., "immediate", "near", "mid", "far"
    float r_min;            // Inner radius (meters)
    float r_max;            // Outer radius (meters)
    float cell_size;        // Cell side length (meters)
    int   grid_width;       // Number of cells along X in this zone's bounding box
    int   grid_height;      // Number of cells along Y in this zone's bounding box
    int   offset_x;         // Cell offset to center coordinates (grid_width / 2)
    int   offset_y;         // Cell offset to center coordinates (grid_height / 2)
    int   base_idx;         // Starting index in the flat cell array
    int   total_cells;      // Total cells in this zone
};

/// GPU-compatible zone config (POD, no std::string)
struct ZoneConfigGPU {
    float r_min;
    float r_max;
    float cell_size;
    int   grid_width;
    int   grid_height;
    int   offset_x;
    int   offset_y;
    int   base_idx;
    int   total_cells;
};

/// Manages resolution zones and cell index computation.
class ZoneManager {
public:
    ZoneManager() = default;

    /// Add a zone. Zones must be added in order of increasing radius.
    void addZone(const std::string& name, float r_min, float r_max, float cell_size);

    /// Finalize zones: compute grid dimensions, offsets, base indices.
    /// Must be called after all addZone() calls and before any lookups.
    void finalize();

    /// Get the zone index for a given radial distance. Returns -1 if out of range.
    int getZoneIndex(float radial_distance) const;

    /// Compute the flat cell index for a point (x, y) in the given zone.
    /// Returns -1 if the point falls outside the zone's grid bounds.
    int getCellIndex(int zone_idx, float x, float y) const;

    /// Total number of cells across all zones.
    int totalCells() const;

    /// Get zone configs (CPU version).
    const std::vector<ZoneConfig>& zones() const { return zones_; }

    /// Get GPU-compatible zone configs (for HIP kernel upload).
    std::vector<ZoneConfigGPU> gpuZoneConfigs() const;

    /// Number of zones.
    int numZones() const { return static_cast<int>(zones_.size()); }

private:
    std::vector<ZoneConfig> zones_;
    bool finalized_ = false;
};

} // namespace foveated_grid
