# =============================================================================
# segmentation_node.py — ROS2 Semantic Segmentation Inference Node (AMD ROCm)
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.sparse_unet import SparseUNet, HAS_SPCONV
from src.model.pointnet2 import PointNet2SemSeg
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


def load_model_config() -> dict:
    """Load model architecture and training settings from config/model_config.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "model_config.yaml"
    if cfg_path.is_file():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


class SegmentationInferenceNode(Node):
    """
    ROS2 Node that subscribes to preprocessed point clouds,
    runs Sparse U-Net semantic segmentation inference on AMD ROCm GPU,
    and publishes classified points (x, y, z, class, confidence).
    """

    def __init__(self, node_name: str = "segmentation_node", model_weights: Optional[str] = None):
        super().__init__(node_name)

        self.cfg = load_model_config()
        self.num_classes = int(self.cfg.get("model", {}).get("num_classes", 6))

        # Device selection: AMD ROCm exposes GPUs through torch.cuda
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            device_name = torch.cuda.get_device_name(0)
            self.get_logger().info(f"Using AMD GPU device: {device_name}")
        else:
            self.device = torch.device("cpu")
            self.get_logger().info("Using CPU device for segmentation")

        # Initialize Sparse U-Net model
        config_path = PROJECT_ROOT / "config" / "model_config.yaml"
        self.model = SparseUNet.from_config(config_path)
        if model_weights is not None and Path(model_weights).is_file():
            self.get_logger().info(f"Loading checkpoint weights: {model_weights}")
            state_dict = torch.load(model_weights, map_location=self.device)
            self.model.load_state_dict(state_dict.get("model_state_dict", state_dict))

        self.model.to(self.device)
        self.model.eval()

        # Voxelizer matching training settings
        voxel_sizes = self.cfg.get("preprocessing", {}).get("voxel_size", [0.05, 0.05, 0.05])
        pc_range = self.cfg.get("preprocessing", {}).get("point_cloud_range", [-100.0, -100.0, -3.0, 100.0, 100.0, 3.0])
        self.voxelizer = Voxelizer(voxel_size=voxel_sizes, point_cloud_range=pc_range)

        # Declare ROS parameters
        self.declare_parameter("input_topic", "/lidar/preprocessed")
        self.declare_parameter("output_topic", "/lidar/classified_points")
        self.declare_parameter("frame_id", "lidar_link")

        in_topic = self.get_parameter("input_topic").value
        out_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        # ROS2 Subscriptions and Publishers
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
            f"SegmentationInferenceNode ready. Subscribing to '{in_topic}', publishing to '{out_topic}'"
        )

    @torch.no_grad()
    def predict_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Run inference on a numpy point cloud (N, 3..5).
        Returns classified points array of shape (N, 5):
          [x, y, z, predicted_class, confidence]
        """
        if len(points) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        # Prepare 4-channel input (x, y, z, intensity)
        if points.shape[1] < 4:
            pad = np.zeros((len(points), 4 - points.shape[1]), dtype=np.float32)
            pts_4d = np.hstack([points[:, :3], pad]).astype(np.float32)
        else:
            pts_4d = points[:, :4].astype(np.float32)

        # Voxelize
        vox_res = self.voxelizer.voxelize(pts_4d)
        if len(vox_res.coordinates) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        sparse_tensor = self.voxelizer.to_sparse_tensor(
            vox_res.voxels,
            vox_res.coordinates,
            device=str(self.device),
        )

        # Forward pass
        logits = self.model(sparse_tensor)
        probs = torch.softmax(logits, dim=-1)
        confs, preds = torch.max(probs, dim=-1)

        # Map back to valid points
        valid_pts = points[vox_res.valid_point_mask]
        valid_voxel_mask = (vox_res.point_to_voxel_idx >= 0)
        valid_pts = valid_pts[valid_voxel_mask]
        pv_idx = vox_res.point_to_voxel_idx[valid_voxel_mask]

        pred_classes = preds.cpu().numpy()[pv_idx].astype(np.float32)
        pred_confs = confs.cpu().numpy()[pv_idx].astype(np.float32)

        classified_points = np.column_stack([
            valid_pts[:, :3],
            pred_classes,
            pred_confs,
        ]).astype(np.float32)

        return classified_points

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        """Process incoming preprocessed PointCloud2, run inference, and publish classified cloud."""
        t0 = time.perf_counter()
        raw_pts = pointcloud2_to_numpy(msg)

        if len(raw_pts) == 0:
            return

        classified = self.predict_point_cloud(raw_pts)

        # Fields: [x, y, z, class, confidence]
        out_msg = numpy_to_pointcloud2(
            classified,
            stamp=msg.header.stamp if hasattr(msg, "header") else None,
            frame_id=self.frame_id,
            field_names=["x", "y", "z", "class", "confidence"],
        )

        if out_msg is not None:
            self.publisher.publish(out_msg)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        # self.get_logger().info(f"Classified {len(classified)} points in {duration_ms:.2f} ms")


def main(args=None):
    if HAS_RCLPY and rclpy is not None:
        rclpy.init(args=args)
        node = SegmentationInferenceNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[segmentation_node] Running in standalone mock mode (rclpy not active)")
        node = SegmentationInferenceNode()
        pts = np.random.uniform(-15, 15, size=(500, 4)).astype(np.float32)
        res = node.predict_point_cloud(pts)
        print(f"[segmentation_node] Predicted {len(res)} points. Class range: [{res[:, 3].min()}, {res[:, 3].max()}]")


if __name__ == "__main__":
    main()
