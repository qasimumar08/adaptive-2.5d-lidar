# =============================================================================
# test_ros2_and_viz.py — Unit Tests for ROS2 Nodes & Visualization Dashboard
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import json
from pathlib import Path
import numpy as np
import pytest

from src.visualization.colormap import Colormap, default_colormap
from src.visualization.rviz_publisher import RVizGridVisualizer
from src.visualization.dashboard import GridDashboard
from src.ros2_nodes.ros_compat import (
    pointcloud2_to_numpy,
    numpy_to_pointcloud2,
    HAS_SENSOR_MSGS,
)
from src.ros2_nodes.lidar_subscriber import LidarPreprocessingNode
from src.ros2_nodes.segmentation_node import SegmentationInferenceNode
from src.ros2_nodes.grid_publisher import GridPublisherNode

try:
    import grid_py
    HAS_GRID_PY = True
except ImportError:
    HAS_GRID_PY = False


# -----------------------------------------------------------------------------
# Colormap Tests
# -----------------------------------------------------------------------------

def test_colormap_colors_and_names():
    cmap = Colormap()
    for cid in range(6):
        rgb = cmap.get_color_rgb(cid)
        assert len(rgb) == 3
        assert all(0 <= val <= 255 for val in rgb)

        rgb_norm = cmap.get_color_rgb(cid, normalized=True)
        assert all(0.0 <= val <= 1.0 for val in rgb_norm)

        bgr = cmap.get_color_bgr(cid)
        assert bgr == (rgb[2], rgb[1], rgb[0])

        rgba = cmap.get_color_rgba(cid, alpha=0.5, normalized=True)
        assert rgba[3] == 0.5

        name = cmap.get_class_name(cid)
        assert isinstance(name, str) and len(name) > 0


def test_colorize_grids():
    cmap = Colormap()
    grid_classes = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ], dtype=np.uint8)

    colored_rgb = cmap.colorize_semantic_grid(grid_classes, bgr=False)
    assert colored_rgb.shape == (2, 3, 3)
    assert colored_rgb.dtype == np.uint8

    colored_bgr = cmap.colorize_semantic_grid(grid_classes, bgr=True)
    assert colored_bgr.shape == (2, 3, 3)

    # Elevation heatmap
    elev_grid = np.array([
        [-1.0, 0.0, 2.5],
        [np.nan, 5.0, -1000.0],
    ], dtype=np.float32)
    heat_rgb = cmap.colorize_elevation_grid(elev_grid, vmin=-2.0, vmax=4.0)
    assert heat_rgb.shape == (2, 3, 3)
    assert heat_rgb.dtype == np.uint8


# -----------------------------------------------------------------------------
# PointCloud2 Conversion Tests
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SENSOR_MSGS, reason="sensor_msgs not available")
def test_pointcloud2_conversions():
    pts_in = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [-4.0, 5.0, -6.0, 0.9],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    msg = numpy_to_pointcloud2(pts_in, frame_id="lidar_link", field_names=["x", "y", "z", "intensity"])
    assert msg is not None
    assert msg.header.frame_id == "lidar_link"
    assert msg.width == 3

    pts_out = pointcloud2_to_numpy(msg)
    np.testing.assert_allclose(pts_in, pts_out, atol=1e-5)


# -----------------------------------------------------------------------------
# RVizGridVisualizer Tests
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GRID_PY, reason="grid_py not built")
def test_rviz_grid_visualizer():
    from tests.test_grid_engine import build_configured_zone_manager

    zm = build_configured_zone_manager()
    grid = grid_py.FoveatedGrid(zm)

    # Project point in immediate zone and near zone
    pts = np.array([
        [2.0, 2.0, 1.0, 0, 0.9],
        [15.0, 0.0, 0.5, 2, 0.85],
    ], dtype=np.float32)
    grid.projectPoints(pts)
    grid.finalizeCells()

    viz = RVizGridVisualizer(frame_id="lidar_link")
    markers = viz.grid_to_marker_array(grid, include_free_surface=True, include_zone_rings=True)

    if hasattr(markers, "markers"):
        # Real MarkerArray
        assert len(markers.markers) >= 2  # At least 1 cube list + 1 ring list
        cube_marker = markers.markers[0]
        assert cube_marker.header.frame_id == "lidar_link"
        assert len(cube_marker.points) > 0
    else:
        # Fallback dict
        assert "zones" in markers
        assert len(markers["zones"]) == zm.numZones()


# -----------------------------------------------------------------------------
# Dashboard Tests
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GRID_PY, reason="grid_py not built")
def test_dashboard_headless_render():
    from tests.test_grid_engine import build_configured_zone_manager

    zm = build_configured_zone_manager()
    grid = grid_py.FoveatedGrid(zm)

    pts = np.array([
        [3.0, 4.0, 0.5, 0, 0.9],
        [20.0, -10.0, 1.2, 3, 0.8],
    ], dtype=np.float32)
    grid.projectPoints(pts)
    grid.finalizeCells()

    dash = GridDashboard(canvas_size=300, max_range=100.0, headless=True)
    canvas = dash.update_frame(grid, latency_ms=15.0)

    assert canvas.shape == (300, 300, 3)
    assert dash.current_fps > 0.0
    assert dash.memory_stats["savings_pct"] > 90.0
    dash.close()


# -----------------------------------------------------------------------------
# LidarPreprocessingNode Tests
# -----------------------------------------------------------------------------

def test_lidar_preprocessing_node():
    node = LidarPreprocessingNode(node_name="test_preprocessor")

    # Synthetic cloud: 200 points
    rng = np.random.default_rng(42)
    xyz = rng.uniform(-10.0, 10.0, size=(200, 3))
    # Place ground plane at z ~ 0.0
    xyz[:100, 2] = rng.normal(0.0, 0.02, size=100)
    intensities = rng.uniform(0.1, 1.0, size=(200, 1))

    raw_pts = np.hstack([xyz, intensities]).astype(np.float32)
    preproc = node.process_point_cloud(raw_pts)

    assert preproc.shape[1] == 5  # [x, y, z, intensity, is_ground]
    assert preproc.shape[0] > 0
    assert np.any(preproc[:, 4] == 1.0)  # Ground points detected


# -----------------------------------------------------------------------------
# SegmentationInferenceNode Tests
# -----------------------------------------------------------------------------

def test_segmentation_inference_node():
    node = SegmentationInferenceNode(node_name="test_segmentation")

    pts = np.array([
        [1.0, 1.0, 0.0, 0.8, 1.0],
        [2.0, 2.0, 0.5, 0.5, 0.0],
        [-3.0, 4.0, 1.0, 0.2, 0.0],
    ], dtype=np.float32)

    classified = node.predict_point_cloud(pts)
    assert classified.shape[1] == 5  # [x, y, z, class, confidence]
    assert classified.shape[0] > 0
    assert all(0 <= c < 6 for c in classified[:, 3])


# -----------------------------------------------------------------------------
# GridPublisherNode Tests
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GRID_PY, reason="grid_py not built")
def test_grid_publisher_node():
    node = GridPublisherNode(node_name="test_grid_publisher")

    classified_pts = np.array([
        [2.0, 2.0, 0.5, 0, 0.95],
        [15.0, 5.0, 1.2, 3, 0.85],
        [45.0, -10.0, 2.0, 2, 0.70],
    ], dtype=np.float32)

    telemetry = node.process_classified_points(classified_pts)

    assert telemetry["input_points"] == 3
    assert telemetry["foveated_cells"] == 910400
    assert telemetry["memory_reduction_pct"] > 90.0
    assert telemetry["projection_latency_ms"] > 0.0
