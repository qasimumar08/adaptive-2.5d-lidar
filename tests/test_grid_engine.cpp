// =============================================================================
// test_grid_engine.cpp — Unit Tests for Foveated Grid Engine (C++ / HIP)
// PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
// =============================================================================

#include <gtest/gtest.h>
#include "cell.h"
#include "zone_manager.h"
#include "foveated_grid.h"
#include "hip_projection.h"
#include <cmath>
#include <vector>

using namespace foveated_grid;

// Helper to construct a standard 4-zone manager matching config/grid_params.yaml
ZoneManager createStandardZoneManager() {
    ZoneManager zm;
    // Zone 1: immediate 0-10m @ 5cm
    zm.addZone("immediate", 0.0f, 10.0f, 0.05f);
    // Zone 2: near 10-30m @ 10cm
    zm.addZone("near", 10.0f, 30.0f, 0.10f);
    // Zone 3: mid 30-60m @ 25cm
    zm.addZone("mid", 30.0f, 60.0f, 0.25f);
    // Zone 4: far 60-100m @ 50cm
    zm.addZone("far", 60.0f, 100.0f, 0.50f);
    zm.finalize();
    return zm;
}

// -----------------------------------------------------------------------------
// ZoneManager Tests
// -----------------------------------------------------------------------------

TEST(ZoneManagerTest, AddZoneValidation) {
    ZoneManager zm;
    // Negative r_min
    EXPECT_THROW(zm.addZone("bad1", -1.0f, 10.0f, 0.1f), std::invalid_argument);
    // r_min >= r_max
    EXPECT_THROW(zm.addZone("bad2", 10.0f, 5.0f, 0.1f), std::invalid_argument);
    EXPECT_THROW(zm.addZone("bad3", 10.0f, 10.0f, 0.1f), std::invalid_argument);
    // Negative or zero cell_size
    EXPECT_THROW(zm.addZone("bad4", 0.0f, 10.0f, 0.0f), std::invalid_argument);
    EXPECT_THROW(zm.addZone("bad5", 0.0f, 10.0f, -0.05f), std::invalid_argument);

    // Non-increasing radii
    zm.addZone("zone1", 0.0f, 10.0f, 0.1f);
    EXPECT_THROW(zm.addZone("zone2", 8.0f, 20.0f, 0.2f), std::invalid_argument);
}

TEST(ZoneManagerTest, FinalizeCalculations) {
    ZoneManager zm = createStandardZoneManager();
    EXPECT_EQ(zm.numZones(), 4);

    const auto& zones = zm.zones();

    // Zone 0: 0-10m @ 0.05m -> 2 * 10 / 0.05 = 400
    EXPECT_EQ(zones[0].grid_width, 400);
    EXPECT_EQ(zones[0].grid_height, 400);
    EXPECT_EQ(zones[0].offset_x, 200);
    EXPECT_EQ(zones[0].offset_y, 200);
    EXPECT_EQ(zones[0].base_idx, 0);
    EXPECT_EQ(zones[0].total_cells, 160000);

    // Zone 1: 10-30m @ 0.10m -> 2 * 30 / 0.10 = 600
    EXPECT_EQ(zones[1].grid_width, 600);
    EXPECT_EQ(zones[1].grid_height, 600);
    EXPECT_EQ(zones[1].offset_x, 300);
    EXPECT_EQ(zones[1].offset_y, 300);
    EXPECT_EQ(zones[1].base_idx, 160000);
    EXPECT_EQ(zones[1].total_cells, 360000);

    // Zone 2: 30-60m @ 0.25m -> 2 * 60 / 0.25 = 480
    EXPECT_EQ(zones[2].grid_width, 480);
    EXPECT_EQ(zones[2].grid_height, 480);
    EXPECT_EQ(zones[2].offset_x, 240);
    EXPECT_EQ(zones[2].offset_y, 240);
    EXPECT_EQ(zones[2].base_idx, 520000);
    EXPECT_EQ(zones[2].total_cells, 230400);

    // Zone 3: 60-100m @ 0.50m -> 2 * 100 / 0.50 = 400
    EXPECT_EQ(zones[3].grid_width, 400);
    EXPECT_EQ(zones[3].grid_height, 400);
    EXPECT_EQ(zones[3].offset_x, 200);
    EXPECT_EQ(zones[3].offset_y, 200);
    EXPECT_EQ(zones[3].base_idx, 750400);
    EXPECT_EQ(zones[3].total_cells, 160000);

    // Total cells: 160000 + 360000 + 230400 + 160000 = 910400
    EXPECT_EQ(zm.totalCells(), 910400);
}

TEST(ZoneManagerTest, GetZoneIndex) {
    ZoneManager zm = createStandardZoneManager();

    EXPECT_EQ(zm.getZoneIndex(0.0f), 0);
    EXPECT_EQ(zm.getZoneIndex(5.0f), 0);
    EXPECT_EQ(zm.getZoneIndex(9.999f), 0);

    EXPECT_EQ(zm.getZoneIndex(10.0f), 1);
    EXPECT_EQ(zm.getZoneIndex(20.0f), 1);
    EXPECT_EQ(zm.getZoneIndex(29.999f), 1);

    EXPECT_EQ(zm.getZoneIndex(30.0f), 2);
    EXPECT_EQ(zm.getZoneIndex(45.0f), 2);
    EXPECT_EQ(zm.getZoneIndex(59.999f), 2);

    EXPECT_EQ(zm.getZoneIndex(60.0f), 3);
    EXPECT_EQ(zm.getZoneIndex(80.0f), 3);
    EXPECT_EQ(zm.getZoneIndex(100.0f), 3); // Outer boundary check

    // Out of bounds
    EXPECT_EQ(zm.getZoneIndex(-0.1f), -1);
    EXPECT_EQ(zm.getZoneIndex(100.1f), -1);
    EXPECT_EQ(zm.getZoneIndex(200.0f), -1);
}

TEST(ZoneManagerTest, GetCellIndex) {
    ZoneManager zm = createStandardZoneManager();

    // Center point (0.0, 0.0) in Zone 0
    int idx0 = zm.getCellIndex(0, 0.0f, 0.0f);
    // gx = floor(0/0.05) + 200 = 200, gy = 200
    // base_idx = 0 -> 200 * 400 + 200 = 80200
    EXPECT_EQ(idx0, 80200);

    // Point (+1.0, +1.0) in Zone 0
    // gx = floor(1.0/0.05) + 200 = 20 + 200 = 220
    // gy = floor(1.0/0.05) + 200 = 220
    int idx1 = zm.getCellIndex(0, 1.0f, 1.0f);
    EXPECT_EQ(idx1, 220 * 400 + 220);

    // Negative coordinates (-1.0, -2.0) in Zone 0
    // gx = floor(-1.0/0.05) + 200 = -20 + 200 = 180
    // gy = floor(-2.0/0.05) + 200 = -40 + 200 = 160
    int idx2 = zm.getCellIndex(0, -1.0f, -2.0f);
    EXPECT_EQ(idx2, 160 * 400 + 180);

    // Invalid zone
    EXPECT_EQ(zm.getCellIndex(-1, 0.0f, 0.0f), -1);
    EXPECT_EQ(zm.getCellIndex(4, 0.0f, 0.0f), -1);

    // Coordinate outside zone bounding box
    EXPECT_EQ(zm.getCellIndex(0, 50.0f, 50.0f), -1);
}

TEST(ZoneManagerTest, GpuZoneConfigs) {
    ZoneManager zm = createStandardZoneManager();
    std::vector<ZoneConfigGPU> gpu_zones = zm.gpuZoneConfigs();

    ASSERT_EQ(gpu_zones.size(), 4);
    EXPECT_FLOAT_EQ(gpu_zones[0].r_min, 0.0f);
    EXPECT_FLOAT_EQ(gpu_zones[0].r_max, 10.0f);
    EXPECT_FLOAT_EQ(gpu_zones[0].cell_size, 0.05f);
    EXPECT_EQ(gpu_zones[0].grid_width, 400);
    EXPECT_EQ(gpu_zones[0].grid_height, 400);
    EXPECT_EQ(gpu_zones[0].offset_x, 200);
    EXPECT_EQ(gpu_zones[0].offset_y, 200);
    EXPECT_EQ(gpu_zones[0].base_idx, 0);
    EXPECT_EQ(gpu_zones[0].total_cells, 160000);

    EXPECT_FLOAT_EQ(gpu_zones[3].r_min, 60.0f);
    EXPECT_FLOAT_EQ(gpu_zones[3].r_max, 100.0f);
    EXPECT_FLOAT_EQ(gpu_zones[3].cell_size, 0.50f);
    EXPECT_EQ(gpu_zones[3].base_idx, 750400);
}

// -----------------------------------------------------------------------------
// FoveatedGrid Tests
// -----------------------------------------------------------------------------

TEST(FoveatedGridTest, InitializationAndClear) {
    ZoneManager zm = createStandardZoneManager();
    FoveatedGrid grid(zm);

    EXPECT_EQ(grid.totalCells(), 910400);
    EXPECT_EQ(grid.cells().size(), 910400);
    EXPECT_EQ(grid.memoryUsage(), 910400 * sizeof(Cell));

    // Verify default cell state
    const auto& c0 = grid.getCell(0, 200, 200);
    EXPECT_EQ(c0.point_count, 0);
    EXPECT_EQ(c0.occupancy, 2);
    EXPECT_EQ(c0.semantic_class, 5);
    EXPECT_FLOAT_EQ(c0.confidence, 0.0f);
    EXPECT_FLOAT_EQ(c0.elevation_mean(), 0.0f);
    EXPECT_FLOAT_EQ(c0.height_span(), 0.0f);
}

TEST(FoveatedGridTest, CPUPointProjectionAndFinalize) {
    ZoneManager zm = createStandardZoneManager();
    FoveatedGrid grid(zm);

    std::vector<ClassifiedPoint> points = {
        // Two points in immediate zone (r ~ 2.82m) in same cell
        {2.01f, 2.01f, 1.5f, 0, 0.9f},  // drivable surface
        {2.02f, 2.03f, 2.5f, 0, 0.8f},  // drivable surface

        // Obstacle point in near zone (r = 15m)
        {15.0f, 0.0f, 0.5f, 2, 0.95f},  // static obstacle

        // Vehicle points in mid zone (r ~ 42.4m)
        {30.0f, 30.0f, 1.0f, 3, 0.85f}, // dynamic vehicle
        {30.1f, 30.1f, 1.2f, 3, 0.75f}, // dynamic vehicle
        {30.05f, 30.05f, 1.1f, 0, 0.60f} // minority vote: drivable
    };

    grid.projectPoints(points);

    // Check pre-finalized cell for the first 2 points
    int c_idx = zm.getCellIndex(0, 2.01f, 2.01f);
    ASSERT_GE(c_idx, 0);
    const Cell& cell0 = grid.cells()[c_idx];
    EXPECT_EQ(cell0.point_count, 2);
    EXPECT_FLOAT_EQ(cell0.elevation_min, 1.5f);
    EXPECT_FLOAT_EQ(cell0.elevation_max, 2.5f);
    EXPECT_FLOAT_EQ(cell0.elevation_sum, 4.0f);
    EXPECT_FLOAT_EQ(cell0.elevation_mean(), 2.0f);
    EXPECT_FLOAT_EQ(cell0.height_span(), 1.0f);

    grid.finalizeCells(0, 0.5f);

    // Cell 0 should be drivable surface (0) and occupancy free (0)
    const Cell& final_c0 = grid.cells()[c_idx];
    EXPECT_EQ(final_c0.semantic_class, 0);
    EXPECT_EQ(final_c0.occupancy, 0); // free
    EXPECT_NEAR(final_c0.confidence, 0.85f, 1e-4f);

    // Obstacle point in near zone
    int c_near = zm.getCellIndex(1, 15.0f, 0.0f);
    ASSERT_GE(c_near, 0);
    const Cell& final_cnear = grid.cells()[c_near];
    EXPECT_EQ(final_cnear.semantic_class, 2);
    EXPECT_EQ(final_cnear.occupancy, 1); // occupied
    EXPECT_FLOAT_EQ(final_cnear.confidence, 0.95f);

    // Vehicle points in mid zone (majority vote: 2 vehicle vs 1 drivable -> vehicle)
    int c_mid = zm.getCellIndex(2, 30.0f, 30.0f);
    ASSERT_GE(c_mid, 0);
    const Cell& final_cmid = grid.cells()[c_mid];
    EXPECT_EQ(final_cmid.point_count, 3);
    EXPECT_EQ(final_cmid.semantic_class, 3); // dynamic vehicle
    EXPECT_EQ(final_cmid.occupancy, 1);      // occupied
    EXPECT_FLOAT_EQ(final_cmid.elevation_mean(), (1.0f + 1.2f + 1.1f) / 3.0f);
}

TEST(FoveatedGridTest, GPUAndCPUProjectionEquivalence) {
    ZoneManager zm = createStandardZoneManager();
    FoveatedGrid grid_cpu(zm);
    FoveatedGrid grid_gpu(zm);

    std::vector<ClassifiedPoint> points = {
        {1.0f, 1.0f, 0.2f, 0, 0.9f},
        {1.02f, 1.01f, 0.4f, 0, 0.8f},
        {-5.0f, 5.0f, -0.5f, 1, 0.7f},
        {12.0f, -8.0f, 1.8f, 2, 0.85f},
        {-25.0f, 20.0f, 2.0f, 3, 0.95f},
        {40.0f, 40.0f, 0.0f, 4, 0.65f},
        {70.0f, -60.0f, 3.5f, 1, 0.75f}
    };

    grid_cpu.projectPoints(points);
    grid_cpu.finalizeCells();

    grid_gpu.projectPointsGPU(points);
    grid_gpu.finalizeCells();

    for (size_t i = 0; i < grid_cpu.cells().size(); ++i) {
        const auto& c_cpu = grid_cpu.cells()[i];
        const auto& c_gpu = grid_gpu.cells()[i];

        EXPECT_EQ(c_cpu.point_count, c_gpu.point_count);
        EXPECT_EQ(c_cpu.semantic_class, c_gpu.semantic_class);
        EXPECT_EQ(c_cpu.occupancy, c_gpu.occupancy);
        EXPECT_FLOAT_EQ(c_cpu.confidence, c_gpu.confidence);
        EXPECT_FLOAT_EQ(c_cpu.elevation_mean(), c_gpu.elevation_mean());
        if (c_cpu.point_count > 0) {
            EXPECT_FLOAT_EQ(c_cpu.elevation_min, c_gpu.elevation_min);
            EXPECT_FLOAT_EQ(c_cpu.elevation_max, c_gpu.elevation_max);
        }
    }
}

TEST(FoveatedGridTest, ConfidenceThresholdAndOccupancy) {
    ZoneManager zm = createStandardZoneManager();
    FoveatedGrid grid(zm);

    std::vector<ClassifiedPoint> points = {
        // Low confidence obstacle (confidence 0.3 < threshold 0.5)
        {15.0f, 0.0f, 1.0f, 2, 0.3f},
        // High confidence obstacle
        {20.0f, 0.0f, 1.0f, 2, 0.8f},
        // Low confidence drivable surface (confidence 0.3)
        {5.0f, 0.0f, 0.0f, 0, 0.3f}
    };

    grid.projectPoints(points);
    grid.finalizeCells(0, 0.5f);

    int idx_low_obs = zm.getCellIndex(1, 15.0f, 0.0f);
    int idx_high_obs = zm.getCellIndex(1, 20.0f, 0.0f);
    int idx_low_drivable = zm.getCellIndex(0, 5.0f, 0.0f);

    // Low confidence obstacle should become occupancy 2 (unknown)
    EXPECT_EQ(grid.cells()[idx_low_obs].occupancy, 2);

    // High confidence obstacle should be occupancy 1 (occupied)
    EXPECT_EQ(grid.cells()[idx_high_obs].occupancy, 1);

    // Drivable surface remains free (0)
    EXPECT_EQ(grid.cells()[idx_low_drivable].occupancy, 0);
}

TEST(FoveatedGridTest, BoundaryEdgeCases) {
    ZoneManager zm = createStandardZoneManager();
    FoveatedGrid grid(zm);

    // Points at exact boundary distances:
    // r = 10.0m is start of zone 1
    // r = 30.0m is start of zone 2
    // r = 60.0m is start of zone 3
    // r = 100.0m is outer edge of zone 3
    std::vector<ClassifiedPoint> points = {
        {10.0f, 0.0f, 0.0f, 0, 1.0f},
        {30.0f, 0.0f, 0.0f, 1, 1.0f},
        {60.0f, 0.0f, 0.0f, 2, 1.0f},
        {100.0f, 0.0f, 0.0f, 3, 1.0f},
        {101.0f, 0.0f, 0.0f, 4, 1.0f} // Out of bounds -> ignored
    };

    grid.projectPoints(points);
    grid.finalizeCells();

    // Verify zone assignments
    int idx10 = zm.getCellIndex(1, 10.0f, 0.0f);
    EXPECT_EQ(grid.cells()[idx10].point_count, 1);

    int idx30 = zm.getCellIndex(2, 30.0f, 0.0f);
    EXPECT_EQ(grid.cells()[idx30].point_count, 1);

    int idx60 = zm.getCellIndex(3, 60.0f, 0.0f);
    EXPECT_EQ(grid.cells()[idx60].point_count, 1);

    int idx100 = zm.getCellIndex(3, 100.0f, 0.0f);
    EXPECT_EQ(grid.cells()[idx100].point_count, 1);
}

TEST(FoveatedGridTest, OutOfBoundsGetCellThrows) {
    ZoneManager zm = createStandardZoneManager();
    FoveatedGrid grid(zm);

    EXPECT_THROW(grid.getCell(-1, 0, 0), std::out_of_range);
    EXPECT_THROW(grid.getCell(4, 0, 0), std::out_of_range);
    EXPECT_THROW(grid.getCell(0, -1, 0), std::out_of_range);
    EXPECT_THROW(grid.getCell(0, 400, 0), std::out_of_range);
    EXPECT_THROW(grid.getCell(0, 0, 400), std::out_of_range);
    EXPECT_NO_THROW(grid.getCell(0, 0, 0));
    EXPECT_NO_THROW(grid.getCell(0, 399, 399));
}
