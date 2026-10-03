"""Ego-motion compensation (scan deskewing) for 3D Lidar point clouds.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
This module provides motion compensation to remove distortion caused by
vehicle ego-motion during the Lidar spinning sweep (~50-100 ms).
Parameters are aligned with config/sensor_config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import logging

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class EgoMotionCompensator:
    """Compensates point clouds for vehicle motion during scan acquisition.

    In spinning Lidars (e.g. Velodyne VLP-16, Ouster OS1), points in a single
    frame are collected sequentially over time. If the vehicle is translating
    or rotating, points become distorted (skewed). This compensator deskews
    points to a common reference time (default: end of scan sweep).
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        enabled: bool = True,
        odom_topic: str = "/odom",
    ) -> None:
        """Initialize ego-motion compensator.

        Args:
            config_path: Optional path to sensor_config.yaml.
            enabled: Master switch for ego-motion compensation.
            odom_topic: ROS2 odometry topic name.
        """
        self.enabled = enabled
        self.odom_topic = odom_topic

        if config_path is not None:
            self.load_config(config_path)

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load ego-motion parameters from YAML configuration.

        Args:
            config_path: Path to sensor_config.yaml.
        """
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            logger.warning("Config path %s not found. Using defaults.", cfg_path)
            return

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        comp_cfg = cfg.get("preprocessing", {}).get("ego_motion_compensation", {})
        if comp_cfg:
            self.enabled = bool(comp_cfg.get("enabled", self.enabled))
            self.odom_topic = str(comp_cfg.get("odom_topic", self.odom_topic))

    def compensate(
        self,
        points: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compensate point cloud for vehicle motion.

        If compensation is disabled or timestamps are unavailable, returns points as-is.

        Args:
            points: Array of shape (N, C) where columns 0..2 are (x, y, z).
            timestamps: Optional array of shape (N,) indicating relative timestamp [0, scan_time]
                for each point, or azimuth angles [0, 2*pi] from which time is inferred.
            linear_velocity: Vehicle translational velocity [vx, vy, vz] in m/s.
            angular_velocity: Vehicle rotational velocity [wx, wy, wz] in rad/s.

        Returns:
            Deskewed point cloud array of shape (N, C).
        """
        if not self.enabled or points is None or len(points) == 0:
            return points

        if timestamps is None or (linear_velocity is None and angular_velocity is None):
            # No odometry / timing telemetry supplied -> passthrough
            return points.copy()

        pts_deskewed = points.copy()
        xyz = pts_deskewed[:, :3].astype(np.float64, copy=False)

        # Normalize relative timestamps to [0, dt]
        t = timestamps.astype(np.float64)
        if t.max() > t.min():
            t_rel = t - t.max()  # Deskew towards end of scan frame (t_rel <= 0)
        else:
            t_rel = np.zeros_like(t)

        lin_v = np.zeros(3, dtype=np.float64) if linear_velocity is None else np.asarray(linear_velocity, dtype=np.float64)
        ang_v = np.zeros(3, dtype=np.float64) if angular_velocity is None else np.asarray(angular_velocity, dtype=np.float64)

        # 1. Translational compensation: delta_p = - v * t_rel
        xyz += np.outer(t_rel, lin_v)

        # 2. Rotational compensation using Rodrigues formula for small sweep angles
        ang_mag = np.linalg.norm(ang_v)
        if ang_mag > 1e-6:
            # Per-point rotation angle: theta = - ang_mag * t_rel
            axis = ang_v / ang_mag
            # For each point, apply Rodrigues rotation around axis
            cos_a = np.cos(t_rel * ang_mag)
            sin_a = np.sin(t_rel * ang_mag)

            # R(t) * p = p*cos(a) + (axis x p)*sin(a) + axis*(axis . p)*(1 - cos(a))
            cross_prod = np.cross(axis, xyz)
            dot_prod = np.sum(axis * xyz, axis=1, keepdims=True)

            xyz = (
                xyz * cos_a[:, np.newaxis]
                + cross_prod * sin_a[:, np.newaxis]
                + axis * dot_prod * (1.0 - cos_a[:, np.newaxis])
            )

        pts_deskewed[:, :3] = xyz.astype(pts_deskewed.dtype)
        return pts_deskewed

    def estimate_timestamps_from_azimuth(
        self, points: np.ndarray, scan_period: float = 0.1
    ) -> np.ndarray:
        """Estimate per-point acquisition timestamps from azimuth angles for spinning Lidars.

        Assumes clockwise spinning around +Z axis starting from -X or +X.

        Args:
            points: (N, C) points where cols 0, 1 are x, y.
            scan_period: Full sweep time in seconds (e.g. 0.1s for 10Hz, 0.05s for 20Hz).

        Returns:
            Timestamps (N,) in seconds in range [0, scan_period].
        """
        if points is None or len(points) == 0:
            return np.empty(0, dtype=np.float32)

        x = points[:, 0]
        y = points[:, 1]
        # Azimuth in [0, 2*pi]
        angles = np.arctan2(y, x)
        angles = np.mod(angles + 2.0 * np.pi, 2.0 * np.pi)

        # Map angle to relative time
        timestamps = (angles / (2.0 * np.pi)) * scan_period
        return timestamps.astype(np.float32)
