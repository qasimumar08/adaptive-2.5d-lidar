# =============================================================================
# test_grid_engine.py — Python Integration Tests for Foveated Grid Engine
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import math
import os
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml

# Ensure build/ directory is in python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

try:
    import grid_py
except ImportError:
    pytest.skip("grid_py module not found in build/. Run build script first.", allow_module_level=True)


def load_grid_config():
    """Load grid parameters from config/grid_params.yaml."""
    config_path = PROJECT_ROOT / "config" / "grid_params.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_configured_zone_manager():
    """Build and finalize a ZoneManager matching config/grid_params.yaml."""
    cfg = load_grid_config()
    zm = grid_py.ZoneManager()
    for z in cfg["grid"]["zones"]:
        zm.addZone(
            name=z["name"],
            r_min=float(z["radius_min"]),
            r_max=float(z["radius_max"]),
            cell_size=float(z["cell_size"]),
        )
    zm.finalize()
    return zm


def test_zone_configuration_matches_yaml():
    """Test that ZoneManager configuration and calculations match config/grid_params.yaml."""
    cfg = load_grid_config()
    zm = build_configured_zone_manager()
    yaml_zones = cfg["grid"]["zones"]

    assert zm.numZones() == len(yaml_zones)
    zones = zm.zones()

    cumulative_cells = 0
    for i, (z_obj, z_yaml) in enumerate(zip(zones, yaml_zones)):
        assert z_obj.name == z_yaml["name"]
        assert pytest.approx(z_obj.r_min) == z_yaml["radius_min"]
        assert pytest.approx(z_obj.r_max) == z_yaml["radius_max"]
        assert pytest.approx(z_obj.cell_size) == z_yaml["cell_size"]

        # Expected bounding box: 2 * r_max / cell_size
        expected_dim = int(math.ceil(2.0 * z_yaml["radius_max"] / z_yaml["cell_size"]))
        assert z_obj.grid_width == expected_dim
        assert z_obj.grid_height == expected_dim
        assert z_obj.offset_x == expected_dim // 2
        assert z_obj.offset_y == expected_dim // 2
        assert z_obj.base_idx == cumulative_cells
        assert z_obj.total_cells == expected_dim * expected_dim

        cumulative_cells += z_obj.total_cells

    assert zm.totalCells() == cumulative_cells
    # 400x400 + 600x600 + 480x480 + 400x400 = 160000 + 360000 + 230400 + 160000 = 910400
    assert zm.totalCells() == 910400


def test_cpu_projection_synthetic_points():
    """Test CPU projection with synthetic points and verify cell statistics and majority voting."""
    zm = build_configured_zone_manager()
    grid = grid_py.FoveatedGrid(zm)

    # Synthetic points in Zone 0 (immediate: 0-10m) and Zone 1 (near: 10-30m)
    # Cell at (2.0, 2.0): two points with class 0 (drivable), elevations 1.0 and 2.0
    # Cell at (-15.0, 0.0): three points (two vehicle=3, one drivable=0), elevations 0.5, 0.7, 0.6
    points = np.array([
        [2.01, 2.01, 1.0, 0, 0.90],
        [2.02, 2.03, 2.0, 0, 0.80],
        [-15.02, 0.01, 0.5, 3, 0.95],
        [-15.03, 0.02, 0.7, 3, 0.85],
        [-15.01, 0.03, 0.6, 0, 0.70],
    ], dtype=np.float32)

    grid.projectPoints(points)

    # Check cell at (2.0, 2.0) before finalization
    c_idx0 = zm.getCellIndex(0, 2.01, 2.01)
    assert c_idx0 >= 0
    cell0 = grid.cells()[c_idx0]
    assert cell0.point_count == 2
    assert pytest.approx(cell0.elevation_min) == 1.0
    assert pytest.approx(cell0.elevation_max) == 2.0
    assert pytest.approx(cell0.elevation_sum) == 3.0
    assert pytest.approx(cell0.elevation_mean()) == 1.5
    assert pytest.approx(cell0.height_span()) == 1.0

    # Finalize cells
    grid.finalizeCells(unknown_threshold=0, confidence_threshold=0.5)

    # Cell 0: class 0 (drivable), occupancy 0 (free), mean confidence (0.9+0.8)/2 = 0.85
    final_cell0 = grid.cells()[c_idx0]
    assert final_cell0.semantic_class == 0
    assert final_cell0.occupancy == 0  # free
    assert pytest.approx(final_cell0.confidence) == 0.85

    # Cell 1 at (-15, 0): majority vote (2 vehicle vs 1 drivable -> vehicle=3)
    c_idx1 = zm.getCellIndex(1, -15.02, 0.01)
    assert c_idx1 >= 0
    final_cell1 = grid.cells()[c_idx1]
    assert final_cell1.point_count == 3
    assert final_cell1.semantic_class == 3  # dynamic vehicle
    assert final_cell1.occupancy == 1       # occupied
    assert pytest.approx(final_cell1.elevation_mean()) == (0.5 + 0.7 + 0.6) / 3.0


def test_gpu_projection_equivalence():
    """Test that projectPointsGPU produces identical output to projectPoints."""
    zm = build_configured_zone_manager()
    grid_cpu = grid_py.FoveatedGrid(zm)
    grid_gpu = grid_py.FoveatedGrid(zm)

    # Deterministic synthetic point cloud across all zones
    rng = np.random.default_rng(seed=42)
    num_pts = 2000

    # Radial distances distributed across [0.5, 99.5]
    radii = rng.uniform(0.5, 99.5, size=num_pts)
    angles = rng.uniform(-np.pi, np.pi, size=num_pts)

    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    zs = rng.uniform(-2.0, 5.0, size=num_pts)
    classes = rng.integers(0, 6, size=num_pts)
    confs = rng.uniform(0.4, 1.0, size=num_pts)

    points = np.column_stack([xs, ys, zs, classes, confs]).astype(np.float32)

    # Project on CPU
    grid_cpu.projectPoints(points)
    grid_cpu.finalizeCells()

    # Project on GPU
    grid_gpu.projectPointsGPU(points)
    grid_gpu.finalizeCells()

    # Compare populated cells across zones using numpy arrays
    populated_count = 0
    for z in range(zm.numZones()):
        count_cpu = grid_cpu.get_point_count_map(z)
        count_gpu = grid_gpu.get_point_count_map(z)
        np.testing.assert_array_equal(count_cpu, count_gpu)

        mask = count_cpu > 0
        populated_count += int(np.sum(mask))

        elev_cpu = grid_cpu.get_elevation_map(z)
        elev_gpu = grid_gpu.get_elevation_map(z)
        np.testing.assert_allclose(elev_cpu[mask], elev_gpu[mask], rtol=1e-4, atol=1e-4)

        sem_cpu = grid_cpu.get_semantic_map(z)
        sem_gpu = grid_gpu.get_semantic_map(z)
        np.testing.assert_array_equal(sem_cpu[mask], sem_gpu[mask])

        occ_cpu = grid_cpu.get_occupancy_map(z)
        occ_gpu = grid_gpu.get_occupancy_map(z)
        np.testing.assert_array_equal(occ_cpu[mask], occ_gpu[mask])

    assert populated_count > 500


def test_memory_usage_calculation():
    """Test memory usage and verify significant memory reduction vs uniform 5cm grid."""
    zm = build_configured_zone_manager()
    grid = grid_py.FoveatedGrid(zm)

    total_cells = grid.totalCells()
    memory_bytes = grid.memoryUsage()

    # sizeof(Cell) in C++ is 24 bytes (or padded)
    cell_size_bytes = memory_bytes // total_cells
    assert cell_size_bytes > 0
    assert memory_bytes == total_cells * cell_size_bytes

    # A uniform 5cm grid across 100m radius bounding box would require:
    # (200m / 0.05m) x (200m / 0.05m) = 4000 x 4000 = 16,000,000 cells
    uniform_cells = (200.0 / 0.05) ** 2
    reduction_pct = (1.0 - (total_cells / uniform_cells)) * 100.0

    # Expect > 90% memory savings
    assert reduction_pct > 90.0
    print(f"\nFoveated grid cells: {total_cells:,} vs Uniform 5cm grid cells: {int(uniform_cells):,}")
    print(f"Memory reduction: {reduction_pct:.2f}% (Memory: {memory_bytes / (1024*1024):.2f} MB)")


def test_zone_boundary_handling():
    """Test points placed exactly at zone boundary distances."""
    zm = build_configured_zone_manager()
    grid = grid_py.FoveatedGrid(zm)

    # Radii boundaries:
    # 0.0m -> zone 0 (immediate)
    # 10.0m -> zone 1 (near)
    # 30.0m -> zone 2 (mid)
    # 60.0m -> zone 3 (far)
    # 100.0m -> zone 3 (outer bound)
    boundary_points = np.array([
        [0.0, 0.0, 0.1, 0, 1.0],     # Zone 0 center
        [10.0, 0.0, 0.2, 1, 1.0],    # Zone 1 inner boundary
        [30.0, 0.0, 0.3, 2, 1.0],    # Zone 2 inner boundary
        [60.0, 0.0, 0.4, 3, 1.0],    # Zone 3 inner boundary
        [100.0, 0.0, 0.5, 4, 1.0],   # Zone 3 outer boundary
        [100.5, 0.0, 0.6, 5, 1.0],   # Out of bounds (> 100m)
        [-100.5, 0.0, 0.7, 5, 1.0],  # Out of bounds (< -100m)
    ], dtype=np.float32)

    grid.projectPoints(boundary_points)
    grid.finalizeCells()

    # Check zone 0
    c0 = grid.getCell(0, zm.zones()[0].offset_x, zm.zones()[0].offset_y)
    assert c0.point_count == 1
    assert c0.semantic_class == 0

    # Check zone 1 at r=10m
    gx1 = int(math.floor(10.0 / 0.10)) + zm.zones()[1].offset_x
    gy1 = zm.zones()[1].offset_y
    c1 = grid.getCell(1, gx1, gy1)
    assert c1.point_count == 1
    assert c1.semantic_class == 1

    # Check zone 2 at r=30m
    gx2 = int(math.floor(30.0 / 0.25)) + zm.zones()[2].offset_x
    gy2 = zm.zones()[2].offset_y
    c2 = grid.getCell(2, gx2, gy2)
    assert c2.point_count == 1
    assert c2.semantic_class == 2

    # Check zone 3 at r=60m
    gx3 = int(math.floor(60.0 / 0.50)) + zm.zones()[3].offset_x
    gy3 = zm.zones()[3].offset_y
    c3 = grid.getCell(3, gx3, gy3)
    assert c3.point_count == 1
    assert c3.semantic_class == 3

    # Check zone 3 at r=100m (outer edge)
    gx100 = zm.zones()[3].grid_width - 1
    c100 = grid.getCell(3, gx100, gy3)
    assert c100.point_count == 1
    assert c100.semantic_class == 4

    # Points outside 100m must have been ignored without crashes
    total_projected_points = sum(int(np.sum(grid.get_point_count_map(z))) for z in range(zm.numZones()))
    assert total_projected_points == 5


def test_edge_cases():
    """Test edge cases: empty cloud, single point, all points in one zone, and out-of-range lookups."""
    zm = build_configured_zone_manager()
    grid = grid_py.FoveatedGrid(zm)

    # 1. Empty point cloud
    empty_pts = np.zeros((0, 5), dtype=np.float32)
    grid.projectPoints(empty_pts)
    grid.finalizeCells()
    for z in range(zm.numZones()):
        assert np.all(grid.get_point_count_map(z) == 0)

    # 2. Single point in zone 0
    single_pt = np.array([[5.0, 0.0, 1.25, 0, 0.9]], dtype=np.float32)
    grid.projectPoints(single_pt)
    grid.finalizeCells()

    assert sum(int(np.sum(grid.get_point_count_map(z))) for z in range(zm.numZones())) == 1
    gx = int(math.floor(5.0 / 0.05)) + zm.zones()[0].offset_x
    c_single = grid.getCell(0, gx, zm.zones()[0].offset_y)
    assert c_single.point_count == 1
    assert pytest.approx(c_single.elevation_mean()) == 1.25
    assert c_single.occupancy == 0  # drivable

    # 3. Clear and test all points in one zone (zone 2: mid 30-60m)
    grid.clear()
    for z in range(zm.numZones()):
        assert np.all(grid.get_point_count_map(z) == 0)

    mid_pts = np.array([
        [35.0, 0.0, 0.5, 2, 0.9],
        [-35.0, 0.0, 0.6, 2, 0.9],
        [0.0, 40.0, 0.7, 2, 0.9],
        [0.0, -40.0, 0.8, 2, 0.9],
    ], dtype=np.float32)
    grid.projectPoints(mid_pts)
    grid.finalizeCells()

    # Verify only zone 2 has points
    assert np.sum(grid.get_point_count_map(0)) == 0
    assert np.sum(grid.get_point_count_map(1)) == 0
    assert np.sum(grid.get_point_count_map(2)) == 4
    assert np.sum(grid.get_point_count_map(3)) == 0

    # 4. Out of range getCell lookup raises exception
    with pytest.raises(IndexError):
        grid.getCell(-1, 0, 0)
    with pytest.raises(IndexError):
        grid.getCell(4, 0, 0)
    with pytest.raises(IndexError):
        grid.getCell(0, 9999, 0)


def test_2d_map_extractors():
    """Test 2D numpy map extractors (elevation, semantic, occupancy, point_count)."""
    zm = build_configured_zone_manager()
    grid = grid_py.FoveatedGrid(zm)

    # Project point in zone 0
    pts = np.array([[2.0, 2.0, 1.5, 0, 0.9]], dtype=np.float32)
    grid.projectPoints(pts)
    grid.finalizeCells()

    z0 = zm.zones()[0]
    elev_map = grid.get_elevation_map(0)
    sem_map = grid.get_semantic_map(0)
    occ_map = grid.get_occupancy_map(0)
    count_map = grid.get_point_count_map(0)

    assert elev_map.shape == (z0.grid_height, z0.grid_width)
    assert sem_map.shape == (z0.grid_height, z0.grid_width)
    assert occ_map.shape == (z0.grid_height, z0.grid_width)
    assert count_map.shape == (z0.grid_height, z0.grid_width)

    gx = int(math.floor(2.0 / 0.05)) + z0.offset_x
    gy = int(math.floor(2.0 / 0.05)) + z0.offset_y

    assert count_map[gy, gx] == 1
    assert pytest.approx(elev_map[gy, gx]) == 1.5
    assert sem_map[gy, gx] == 0
    assert occ_map[gy, gx] == 0

    # Unobserved cell should be NaN in elevation and 0 in point count
    assert np.isnan(elev_map[0, 0])
    assert count_map[0, 0] == 0
