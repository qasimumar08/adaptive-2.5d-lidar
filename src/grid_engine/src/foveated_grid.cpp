// =============================================================================
// foveated_grid.cpp — Foveated 2.5D Grid Engine Implementation
// Foveated 2.5D Grid Engine — PS 26053
// =============================================================================

#include "foveated_grid.h"
#include "hip_projection.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace foveated_grid {

FoveatedGrid::FoveatedGrid(const ZoneManager& zone_manager)
    : zone_manager_(zone_manager) {
    const int total = zone_manager_.totalCells();
    cells_.resize(total);
    class_votes_.resize(total);
    conf_sums_.resize(total);
    clear();
}

void FoveatedGrid::clear() {
    if (modified_indices_.empty()) {
        for (auto& c : cells_) {
            c.reset();
        }
        for (auto& votes : class_votes_) {
            votes.fill(0);
        }
        std::fill(conf_sums_.begin(), conf_sums_.end(), 0.0f);
    } else {
        for (int idx : modified_indices_) {
            cells_[idx].reset();
            class_votes_[idx].fill(0);
            conf_sums_[idx] = 0.0f;
        }
        modified_indices_.clear();
    }
}

void FoveatedGrid::projectPoints(const std::vector<ClassifiedPoint>& points) {
    for (const auto& p : points) {
        float r = std::sqrt(p.x * p.x + p.y * p.y);
        int z_idx = zone_manager_.getZoneIndex(r);
        if (z_idx < 0) continue;

        int c_idx = zone_manager_.getCellIndex(z_idx, p.x, p.y);
        if (c_idx < 0 || c_idx >= static_cast<int>(cells_.size())) continue;

        Cell& cell = cells_[c_idx];
        if (cell.point_count == 0) {
            modified_indices_.push_back(c_idx);
        }
        if (p.z < cell.elevation_min) cell.elevation_min = p.z;
        if (p.z > cell.elevation_max) cell.elevation_max = p.z;
        cell.elevation_sum += p.z;
        cell.point_count++;

        if (p.semantic_class < 6) {
            class_votes_[c_idx][p.semantic_class]++;
        }
        conf_sums_[c_idx] += p.confidence;
    }
}


void FoveatedGrid::projectPointsGPU(const std::vector<ClassifiedPoint>& points) {
    std::vector<CellGPU> gpu_cells;
    bool success = launchHipProjection(points, zone_manager_.gpuZoneConfigs(), gpu_cells);

    if (!success) {
        std::cerr << "[FoveatedGrid] HIP GPU projection failed. Falling back to CPU projection." << std::endl;
        projectPoints(points);
        return;
    }

    // Merge GPU projection results into grid cells
    for (size_t i = 0; i < cells_.size() && i < gpu_cells.size(); ++i) {
        const auto& gc = gpu_cells[i];
        if (gc.point_count > 0) {
            Cell& cell = cells_[i];
            if (gc.elevation_min < cell.elevation_min) cell.elevation_min = gc.elevation_min;
            if (gc.elevation_max > cell.elevation_max) cell.elevation_max = gc.elevation_max;
            cell.elevation_sum += gc.elevation_sum;
            cell.point_count += gc.point_count;

            for (int c = 0; c < 6; ++c) {
                class_votes_[i][c] += gc.semantic_votes[c];
            }
            conf_sums_[i] += gc.confidence_sum;
        }
    }
}

void FoveatedGrid::finalizeCells(int unknown_threshold, float confidence_threshold) {
    if (modified_indices_.empty()) {
        for (size_t i = 0; i < cells_.size(); ++i) {
            if (cells_[i].point_count > unknown_threshold) {
                modified_indices_.push_back(static_cast<int>(i));
            }
        }
    }

    for (int i : modified_indices_) {
        Cell& cell = cells_[i];

        if (cell.point_count <= unknown_threshold) {
            cell.occupancy = 2; // unknown
            cell.semantic_class = 5; // unknown/noise
            cell.confidence = 0.0f;
            continue;
        }

        // Majority vote for semantic class
        int max_votes = -1;
        int best_class = 5;
        for (int c = 0; c < 6; ++c) {
            if (class_votes_[i][c] > max_votes) {
                max_votes = class_votes_[i][c];
                best_class = c;
            }
        }

        cell.semantic_class = static_cast<uint8_t>(best_class);
        cell.confidence = conf_sums_[i] / cell.point_count;

        // Determine occupancy state:
        // class 0: drivable surface -> free (0)
        // class 1..4: terrain/obstacle/vehicle/pedestrian -> occupied (1)
        // class 5: unknown/noise -> unknown (2)
        if (cell.confidence < confidence_threshold && cell.semantic_class != 0) {
            cell.occupancy = 2; // Below confidence threshold -> unknown
        } else if (cell.semantic_class == 0) {
            cell.occupancy = 0; // free
        } else if (cell.semantic_class >= 1 && cell.semantic_class <= 4) {
            cell.occupancy = 1; // occupied
        } else {
            cell.occupancy = 2; // unknown
        }
    }
}


const Cell& FoveatedGrid::getCell(int zone_idx, int grid_x, int grid_y) const {
    if (zone_idx < 0 || zone_idx >= zone_manager_.numZones()) {
        throw std::out_of_range("Zone index out of range: " + std::to_string(zone_idx));
    }

    const auto& z = zone_manager_.zones()[zone_idx];
    if (grid_x < 0 || grid_x >= z.grid_width || grid_y < 0 || grid_y >= z.grid_height) {
        throw std::out_of_range(
            "Grid position (" + std::to_string(grid_x) + ", " + std::to_string(grid_y) +
            ") out of zone bounds (" + std::to_string(z.grid_width) + ", " +
            std::to_string(z.grid_height) + ")"
        );
    }

    int idx = z.base_idx + grid_y * z.grid_width + grid_x;
    if (idx < 0 || idx >= static_cast<int>(cells_.size())) {
        throw std::out_of_range("Cell index out of range: " + std::to_string(idx));
    }

    return cells_[idx];
}

} // namespace foveated_grid
