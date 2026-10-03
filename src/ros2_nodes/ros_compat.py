# =============================================================================
# ros_compat.py — ROS2 / rclpy Compatibility & PointCloud2 Conversion Utilities
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import sys
import struct
from typing import Any, Callable, List, Optional, Tuple
import numpy as np

# Try importing rclpy
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
    HAS_RCLPY = True
except (ImportError, AttributeError):
    HAS_RCLPY = False
    rclpy = None

    class Node:  # type: ignore
        """Mock ROS2 Node for non-rclpy environments or offline testing."""
        def __init__(self, node_name: str, **kwargs):
            self.node_name = node_name
            self._parameters = {}
            self._publishers = []
            self._subscriptions = []
            self.get_logger = lambda: self

        def info(self, msg: str):
            print(f"[{self.node_name}] [INFO] {msg}")

        def warn(self, msg: str):
            print(f"[{self.node_name}] [WARN] {msg}")

        def error(self, msg: str):
            print(f"[{self.node_name}] [ERROR] {msg}", file=sys.stderr)

        def declare_parameter(self, name: str, value: Any):
            self._parameters[name] = value

        def get_parameter(self, name: str):
            class _Param:
                def __init__(self, v): self.value = v
            return _Param(self._parameters.get(name, None))

        def create_publisher(self, msg_type: Any, topic: str, qos_profile: Any = 10):
            pub = MockPublisher(topic, msg_type)
            self._publishers.append(pub)
            return pub

        def create_subscription(self, msg_type: Any, topic: str, callback: Callable, qos_profile: Any = 10):
            sub = MockSubscription(topic, msg_type, callback)
            self._subscriptions.append(sub)
            return sub

    class MockPublisher:
        def __init__(self, topic: str, msg_type: Any):
            self.topic = topic
            self.msg_type = msg_type
            self.last_published_msg = None
            self.publish_count = 0

        def publish(self, msg: Any):
            self.last_published_msg = msg
            self.publish_count += 1

    class MockSubscription:
        def __init__(self, topic: str, msg_type: Any, callback: Callable):
            self.topic = topic
            self.msg_type = msg_type
            self.callback = callback

    class QoSProfile:  # type: ignore
        def __init__(self, **kwargs): pass

    qos_profile_sensor_data = QoSProfile()


# Import ROS2 message types
try:
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import sensor_msgs_py.point_cloud2 as pc2
    # Test if message can be populated without C-extension error
    _test = PointCloud2()
    _test.data = b"\x00" * 16
    HAS_SENSOR_MSGS = True
except Exception:
    pc2 = None

    class Header:
        def __init__(self, frame_id: str = "lidar_link", stamp=None):
            self.frame_id = frame_id
            self.stamp = stamp

    class PointField:
        FLOAT32 = 7
        def __init__(self, name: str = "", offset: int = 0, datatype: int = 7, count: int = 1):
            self.name = name
            self.offset = offset
            self.datatype = datatype
            self.count = count

    class PointCloud2:
        def __init__(self):
            self.header = Header()
            self.height = 1
            self.width = 0
            self.fields = []
            self.is_bigendian = False
            self.point_step = 0
            self.row_step = 0
            self.data = b""
            self.is_dense = True

    HAS_SENSOR_MSGS = True


def pointcloud2_to_numpy(msg: Any) -> np.ndarray:
    """
    Extract (x, y, z, intensity) from a sensor_msgs.msg.PointCloud2 into a NumPy float32 array.
    Shape: (N, 4) or (N, 3) if intensity is not present.
    """
    if not HAS_SENSOR_MSGS or msg is None:
        return np.zeros((0, 4), dtype=np.float32)

    # Use fast sensor_msgs_py if available
    try:
        fields = [f.name for f in msg.fields]
        target_fields = [f for f in ["x", "y", "z", "intensity"] if f in fields]
        points = pc2.read_points_numpy(msg, field_names=target_fields)

        # Ensure shape (N, 4) with default intensity = 0.0 if not present
        if "intensity" not in fields and points.shape[1] == 3:
            intensities = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack([points, intensities])

        return points.astype(np.float32)
    except Exception:
        # Fallback raw byte buffer parser
        point_step = msg.point_step
        if point_step == 0 or len(msg.data) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        num_points = len(msg.data) // point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)

        field_offsets = {f.name: f.offset for f in msg.fields}
        off_x = field_offsets.get("x", 0)
        off_y = field_offsets.get("y", 4)
        off_z = field_offsets.get("z", 8)
        off_i = field_offsets.get("intensity", None)

        pts = np.zeros((num_points, 4), dtype=np.float32)
        # Vectorized stride view if standard XYZ layout
        if off_x == 0 and off_y == 4 and off_z == 8 and point_step >= 12:
            stride_view = np.ndarray(
                shape=(num_points, point_step // 4),
                dtype=np.float32,
                buffer=msg.data
            )
            pts[:, 0] = stride_view[:, 0]
            pts[:, 1] = stride_view[:, 1]
            pts[:, 2] = stride_view[:, 2]
            if off_i is not None and off_i % 4 == 0:
                pts[:, 3] = stride_view[:, off_i // 4]
        return pts


def numpy_to_pointcloud2(
    points: np.ndarray,
    stamp: Any = None,
    frame_id: str = "lidar_link",
    field_names: Optional[List[str]] = None,
) -> Any:
    """
    Convert a NumPy array (N, C) into a sensor_msgs.msg.PointCloud2 message.
    Default fields for (N, 4): ['x', 'y', 'z', 'intensity']
    Default fields for (N, 5): ['x', 'y', 'z', 'class', 'confidence']
    """
    if not HAS_SENSOR_MSGS:
        return None

    pts = np.asarray(points, dtype=np.float32)
    N, C = pts.shape if pts.ndim == 2 else (0, 0)

    if field_names is None:
        if C == 3:
            field_names = ["x", "y", "z"]
        elif C == 4:
            field_names = ["x", "y", "z", "intensity"]
        elif C >= 5:
            field_names = ["x", "y", "z", "class", "confidence"]
        else:
            field_names = [f"f_{i}" for i in range(C)]

    header = Header()
    header.frame_id = frame_id
    if stamp is not None:
        header.stamp = stamp

    fields = []
    offset = 0
    for fname in field_names:
        field = PointField()
        field.name = fname
        field.offset = offset
        field.datatype = PointField.FLOAT32
        field.count = 1
        fields.append(field)
        offset += 4

    cloud = PointCloud2()
    cloud.header = header
    cloud.height = 1
    cloud.width = N
    cloud.fields = fields
    cloud.is_bigendian = False
    cloud.point_step = offset
    cloud.row_step = cloud.point_step * N
    cloud.data = pts[:, :len(fields)].tobytes()
    cloud.is_dense = True

    return cloud
