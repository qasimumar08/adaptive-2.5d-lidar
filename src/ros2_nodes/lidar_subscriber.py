# =============================================================================
# lidar_subscriber.py — ROS2 PointCloud2 Preprocessing Node
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

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

from src.preprocessing.ground_removal import RANSACGroundRemoval
from src.preprocessing.voxelizer import Voxelizer
from src.ros2_nodes.ros_compat import (
    HAS_RCLPY,
    Node,
    PointCloud2,
    numpy_to_pointcloud2,
    pointcloud2_to_numpy,
    qos_profile_sensor_data,
    rclpy,
)


def load_sensor_config() -> dict:
    """Load sensor parameters from config/sensor_config.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "sensor_config.yaml"
    if cfg_path.is_file():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


class LidarPreprocessingNode(Node):
    """
    ROS2 Node that ingests raw PointCloud2 scans from Lidar,
    filters noise/range, separates ground surface via RANSAC, voxelizes,
    and publishes the preprocessed point cloud.
    """

    def __init__(self, node_name: str = "lidar_preprocessor"):
        super().__init__(node_name)

        # 1. Load sensor parameters
        sensor_cfg = load_sensor_config()
        active_sensor = sensor_cfg.get("active_sensor", "vlp16")
        sensor_params = sensor_cfg.get(active_sensor, {})
        preproc_cfg = sensor_cfg.get("preprocessing", {})

        default_topic = sensor_params.get("ros2_topic", "/velodyne_points")
        self.frame_id = sensor_params.get("frame_id", "lidar_link")

        # Range filter
        range_cfg = preproc_cfg.get("range_filter", {})
        self.min_range = float(range_cfg.get("min_range", 0.5))
        self.max_range = float(range_cfg.get("max_range", 100.0))

        # Ground removal
        ground_cfg = preproc_cfg.get("ground_removal", {})
        self.ransac_dist = float(ground_cfg.get("ransac_threshold", 0.15))
        self.ransac_iters = int(ground_cfg.get("ransac_iterations", 100))
        self.ground_remover = RANSACGroundRemoval(
            ransac_threshold=self.ransac_dist,
            ransac_iterations=self.ransac_iters,
        )

        # Voxel downsampling
        voxel_cfg = preproc_cfg.get("voxel_downsampling", {})
        self.voxel_size = float(voxel_cfg.get("voxel_size", 0.05))
        self.voxelizer = Voxelizer(
            voxel_size=[self.voxel_size, self.voxel_size, self.voxel_size],
            point_cloud_range=[-self.max_range, -self.max_range, -3.0, self.max_range, self.max_range, 3.0],
        )

        # Declare ROS parameters for dynamic override
        self.declare_parameter("input_topic", default_topic)
        self.declare_parameter("output_topic", "/lidar/preprocessed")
        self.declare_parameter("publish_ground_flag", True)

        in_topic = self.get_parameter("input_topic").value
        out_topic = self.get_parameter("output_topic").value

        # ROS2 Subscription & Publisher
        self.subscription = self.create_subscription(
            PointCloud2,
            in_topic,
            self.pointcloud_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self.publisher = self.create_publisher(
            PointCloud2,
            out_topic,
            qos_profile=10,
        )

        self.get_logger().info(
            f"LidarPreprocessingNode initialized. Subscribing to '{in_topic}', publishing to '{out_topic}'"
        )

    def process_point_cloud(self, raw_points: np.ndarray) -> np.ndarray:
        """
        Pure NumPy preprocessing pipeline (testable offline).
        1. Range filtering [min_range, max_range]
        2. RANSAC ground segmentation
        3. Returns points with ground indicator feature: [x, y, z, intensity, is_ground]
        """
        if len(raw_points) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        # Range filter
        dists = np.sqrt(raw_points[:, 0] ** 2 + raw_points[:, 1] ** 2)
        valid_mask = (dists >= self.min_range) & (dists <= self.max_range) & np.isfinite(raw_points[:, 2])
        filtered_pts = raw_points[valid_mask]

        if len(filtered_pts) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        # Ground plane segmentation
        result = self.ground_remover.segment_ground(filtered_pts)
        is_ground = result.is_ground.astype(np.float32).reshape(-1, 1)

        intensity = filtered_pts[:, 3:4] if filtered_pts.shape[1] >= 4 else np.zeros((len(filtered_pts), 1), dtype=np.float32)
        preprocessed = np.hstack([filtered_pts[:, :3], intensity, is_ground]).astype(np.float32)
        return preprocessed

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        """ROS2 message callback on raw Lidar PointCloud2."""
        t_start = time.perf_counter()
        raw_pts = pointcloud2_to_numpy(msg)

        if len(raw_pts) == 0:
            return

        preprocessed_pts = self.process_point_cloud(raw_pts)

        # Publish preprocessed cloud: fields [x, y, z, intensity, is_ground]
        field_names = ["x", "y", "z", "intensity", "is_ground"]
        out_msg = numpy_to_pointcloud2(
            preprocessed_pts,
            stamp=msg.header.stamp if hasattr(msg, "header") else None,
            frame_id=self.frame_id,
            field_names=field_names,
        )

        if out_msg is not None:
            self.publisher.publish(out_msg)

        duration_ms = (time.perf_counter() - t_start) * 1000.0
        # self.get_logger().info(f"Processed {len(raw_pts)} -> {len(preprocessed_pts)} points in {duration_ms:.2f} ms")


def main(args=None):
    if HAS_RCLPY and rclpy is not None:
        rclpy.init(args=args)
        node = LidarPreprocessingNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[lidar_subscriber] Running in standalone mock mode (rclpy not active)")
        node = LidarPreprocessingNode()
        # Test with synthetic scan
        pts = np.random.uniform(-10, 10, size=(1000, 4)).astype(np.float32)
        res = node.process_point_cloud(pts)
        print(f"[lidar_subscriber] Preprocessed {len(res)} synthetic points.")


if __name__ == "__main__":
    main()
