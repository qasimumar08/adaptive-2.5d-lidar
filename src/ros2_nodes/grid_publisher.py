# =============================================================================
# grid_publisher.py — ROS2 Foveated Grid Publisher & RViz2 Node
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BUILD_DIR = PROJECT_ROOT / "build"
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

try:
    import grid_py
except ImportError:
    grid_py = None

from src.ros2_nodes.ros_compat import (
    HAS_RCLPY,
    Node,
    PointCloud2,
    pointcloud2_to_numpy,
    qos_profile_sensor_data,
    rclpy,
)
from src.visualization.colormap import Colormap, default_colormap
from src.visualization.rviz_publisher import RVizGridVisualizer

try:
    from std_msgs.msg import String as StringMsg
    from visualization_msgs.msg import MarkerArray
    HAS_MSGS = True
except ImportError:
    HAS_MSGS = False
    StringMsg = None
    MarkerArray = None


def load_grid_config() -> dict:
    """Load grid parameters from config/grid_params.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "grid_params.yaml"
    if cfg_path.is_file():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


class GridPublisherNode(Node):
    """
    ROS2 Node that ingests classified points from semantic segmentation,
    projects them into the 2.5D FoveatedGrid via AMD ROCm HIP GPU kernels,
    finalizes cells (majority voting & occupancy), and publishes:
      1. RViz2 MarkerArray for real-time 3D visualization (/grid/markers)
      2. Performance and memory telemetry metrics (/grid/metrics)
    """

    def __init__(self, node_name: str = "grid_publisher_node"):
        super().__init__(node_name)

        if grid_py is None:
            raise RuntimeError("grid_py C++/HIP module not found. Build with ./scripts/build_grid_engine.sh first.")

        # 1. Load config and configure zones
        self.grid_cfg = load_grid_config()
        self.zone_manager = grid_py.ZoneManager()

        for z in self.grid_cfg.get("grid", {}).get("zones", []):
            self.zone_manager.addZone(
                name=str(z["name"]),
                r_min=float(z["radius_min"]),
                r_max=float(z["radius_max"]),
                cell_size=float(z["cell_size"]),
            )
        self.zone_manager.finalize()

        # 2. Instantiate FoveatedGrid
        self.grid = grid_py.FoveatedGrid(self.zone_manager)
        self.total_cells = self.grid.totalCells()

        # Cell thresholds
        cell_cfg = self.grid_cfg.get("grid", {}).get("cell", {})
        self.unknown_thresh = int(cell_cfg.get("unknown_threshold", 0))
        self.conf_thresh = float(cell_cfg.get("confidence_threshold", 0.5))

        # 3. RViz visualizer
        self.frame_id = self.grid_cfg.get("grid", {}).get("frame_id", "lidar_link")
        self.visualizer = RVizGridVisualizer(
            frame_id=self.frame_id,
            colormap=default_colormap,
            min_points_threshold=1,
        )

        # 4. Telemetry tracking
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        # Declare ROS parameters
        self.declare_parameter("input_topic", "/lidar/classified_points")
        self.declare_parameter("marker_topic", "/grid/markers")
        self.declare_parameter("metrics_topic", "/grid/metrics")
        self.declare_parameter("use_gpu_projection", True)

        in_topic = self.get_parameter("input_topic").value
        marker_topic = self.get_parameter("marker_topic").value
        metrics_topic = self.get_parameter("metrics_topic").value
        self.use_gpu = bool(self.get_parameter("use_gpu_projection").value)

        # ROS2 Subscriptions and Publishers
        self.subscription = self.create_subscription(
            PointCloud2,
            in_topic,
            self.classified_points_callback,
            qos_profile=qos_profile_sensor_data,
        )

        if HAS_MSGS:
            self.marker_publisher = self.create_publisher(MarkerArray, marker_topic, qos_profile=10)
            self.metrics_publisher = self.create_publisher(StringMsg, metrics_topic, qos_profile=10)
        else:
            self.marker_publisher = self.create_publisher(None, marker_topic, qos_profile=10)
            self.metrics_publisher = self.create_publisher(None, metrics_topic, qos_profile=10)

        self.get_logger().info(
            f"GridPublisherNode initialized. Total zones: {self.zone_manager.numZones()}, "
            f"Total cells: {self.total_cells:,}. Publishing RViz markers to '{marker_topic}'"
        )

    def process_classified_points(self, points: np.ndarray) -> dict:
        """
        Process classified points into the grid and return telemetry dict (testable offline).
        """
        t0 = time.perf_counter()

        # Reset grid cells for new frame
        self.grid.clear()

        # Project points (GPU path or CPU fallback)
        if len(points) > 0:
            if self.use_gpu:
                self.grid.projectPointsGPU(points)
            else:
                self.grid.projectPoints(points)

            # Majority vote and occupancy finalization
            self.grid.finalizeCells(
                unknown_threshold=self.unknown_thresh,
                confidence_threshold=self.conf_thresh,
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0

        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        self.fps = 1.0 / max(dt, 1e-4)
        self.frame_count += 1

        # Memory calculations
        mem_bytes = self.grid.memoryUsage()
        mem_mb = mem_bytes / (1024.0 * 1024.0)

        # Uniform 5cm 100m grid comparison: 16,000,000 cells
        uniform_cells = (200.0 / 0.05) ** 2
        uniform_mb = (uniform_cells * 24) / (1024.0 * 1024.0)
        savings_pct = (1.0 - (self.total_cells / uniform_cells)) * 100.0

        telemetry = {
            "frame_id": self.frame_count,
            "fps": float(self.fps),
            "projection_latency_ms": float(duration_ms),
            "input_points": int(len(points)),
            "foveated_cells": int(self.total_cells),
            "foveated_memory_mb": float(mem_mb),
            "uniform_5cm_memory_mb": float(uniform_mb),
            "memory_reduction_pct": float(savings_pct),
        }
        return telemetry

    def classified_points_callback(self, msg: PointCloud2) -> None:
        """Handle incoming classified points message."""
        raw_pts = pointcloud2_to_numpy(msg)

        # Project and finalize grid
        telemetry = self.process_classified_points(raw_pts)

        stamp = msg.header.stamp if hasattr(msg, "header") else None

        # 1. Publish RViz2 MarkerArray
        marker_array = self.visualizer.grid_to_marker_array(
            self.grid,
            stamp=stamp,
            include_free_surface=True,
            include_zone_rings=True,
        )

        if marker_array is not None and self.marker_publisher is not None:
            self.marker_publisher.publish(marker_array)

        # 2. Publish Telemetry String
        if self.metrics_publisher is not None:
            if HAS_MSGS and StringMsg is not None:
                smsg = StringMsg()
                smsg.data = json.dumps(telemetry)
                self.metrics_publisher.publish(smsg)
            else:
                self.metrics_publisher.publish(telemetry)


def main(args=None):
    if HAS_RCLPY and rclpy is not None:
        rclpy.init(args=args)
        node = GridPublisherNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[grid_publisher] Running in standalone mock mode (rclpy not active)")
        node = GridPublisherNode()
        pts = np.array([
            [2.0, 2.0, 1.0, 0, 0.9],
            [15.0, 0.0, 0.5, 3, 0.85],
            [45.0, 10.0, 1.2, 2, 0.75],
        ], dtype=np.float32)
        metrics = node.process_classified_points(pts)
        print(f"[grid_publisher] Metrics: {metrics}")


if __name__ == "__main__":
    main()
