// =============================================================================
// zone_manager.cpp — Multi-Resolution Zone Configuration Implementation
// Foveated 2.5D Grid Engine — PS 26053
// =============================================================================

#include "zone_manager.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace foveated_grid {

void ZoneManager::addZone(const std::string& name, float r_min, float r_max, float cell_size) {
    if (r_min < 0.0f || r_min >= r_max) {
        throw std::invalid_argument(
            "Invalid zone radii: r_min (" + std::to_string(r_min) +
            ") must be >= 0 and < r_max (" + std::to_string(r_max) + ")"
        );
    }

    if (cell_size <= 0.0f) {
        throw std::invalid_argument("Cell size must be strictly positive: " + std::to_string(cell_size));
    }

    if (!zones_.empty()) {
        float prev_r_max = zones_.back().r_max;
        if (r_min < prev_r_max - 1e-5f) {
            throw std::invalid_argument(
                "Zones must be added in order of increasing radii. Zone '" + name +
                "' r_min (" + std::to_string(r_min) + ") overlaps previous r_max (" +
                std::to_string(prev_r_max) + ")"
            );
        }
    }

    ZoneConfig config;
    config.name = name;
    config.r_min = r_min;
    config.r_max = r_max;
    config.cell_size = cell_size;
    config.grid_width = 0;
    config.grid_height = 0;
    config.offset_x = 0;
    config.offset_y = 0;
    config.base_idx = 0;
    config.total_cells = 0;

    zones_.push_back(config);
    finalized_ = false;
}

void ZoneManager::finalize() {
    int running_offset = 0;

    for (auto& zone : zones_) {
        // Compute bounding box dimensions covering radius [-r_max, +r_max]
        zone.grid_width = static_cast<int>(std::ceil((2.0f * zone.r_max) / zone.cell_size));
        zone.grid_height = static_cast<int>(std::ceil((2.0f * zone.r_max) / zone.cell_size));

        // Center offsets so point (0, 0) maps to center cell
        zone.offset_x = zone.grid_width / 2;
        zone.offset_y = zone.grid_height / 2;

        zone.base_idx = running_offset;
        zone.total_cells = zone.grid_width * zone.grid_height;

        running_offset += zone.total_cells;
    }

    finalized_ = true;
}

int ZoneManager::getZoneIndex(float radial_distance) const {
    if (radial_distance < 0.0f || zones_.empty()) {
        return -1;
    }

    if (!finalized_) {
        const_cast<ZoneManager*>(this)->finalize();
    }

    for (size_t i = 0; i < zones_.size(); ++i) {
        if (radial_distance >= zones_[i].r_min && radial_distance < zones_[i].r_max) {
            return static_cast<int>(i);
        }
    }

    // Edge check: point exactly on outer boundary of last zone
    if (std::abs(radial_distance - zones_.back().r_max) < 1e-4f) {
        return static_cast<int>(zones_.size()) - 1;
    }

    return -1;
}

int ZoneManager::getCellIndex(int zone_idx, float x, float y) const {
    if (zone_idx < 0 || zone_idx >= static_cast<int>(zones_.size())) {
        return -1;
    }

    if (!finalized_) {
        const_cast<ZoneManager*>(this)->finalize();
    }

    const auto& z = zones_[zone_idx];

    int gx = static_cast<int>(std::floor(x / z.cell_size)) + z.offset_x;
    int gy = static_cast<int>(std::floor(y / z.cell_size)) + z.offset_y;

    if (gx == z.grid_width && std::abs(x - z.r_max) < 1e-4f) {
        gx = z.grid_width - 1;
    }
    if (gy == z.grid_height && std::abs(y - z.r_max) < 1e-4f) {
        gy = z.grid_height - 1;
    }

    if (gx < 0 || gx >= z.grid_width || gy < 0 || gy >= z.grid_height) {
        return -1;
    }

    return z.base_idx + gy * z.grid_width + gx;
}

int ZoneManager::totalCells() const {
    if (zones_.empty()) {
        return 0;
    }

    if (!finalized_) {
        const_cast<ZoneManager*>(this)->finalize();
    }

    const auto& last = zones_.back();
    return last.base_idx + last.total_cells;
}

std::vector<ZoneConfigGPU> ZoneManager::gpuZoneConfigs() const {
    if (!finalized_) {
        const_cast<ZoneManager*>(this)->finalize();
    }

    std::vector<ZoneConfigGPU> gpu_configs;
    gpu_configs.reserve(zones_.size());

    for (const auto& z : zones_) {
        ZoneConfigGPU pod;
        pod.r_min = z.r_min;
        pod.r_max = z.r_max;
        pod.cell_size = z.cell_size;
        pod.grid_width = z.grid_width;
        pod.grid_height = z.grid_height;
        pod.offset_x = z.offset_x;
        pod.offset_y = z.offset_y;
        pod.base_idx = z.base_idx;
        pod.total_cells = z.total_cells;
        gpu_configs.push_back(pod);
    }

    return gpu_configs;
}

} // namespace foveated_grid
