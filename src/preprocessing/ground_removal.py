"""RANSAC-based ground plane removal for 3D Lidar point clouds.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
This module separates ground points from obstacle/terrain points using
RANSAC plane segmentation (via Open3D with a NumPy fallback).
Parameters are aligned with config/sensor_config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import logging

import numpy as np
import yaml

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False


@dataclass
class GroundRemovalResult:
    """Container for ground removal outputs.

    Attributes:
        non_ground_points: Point cloud array (M, C) excluding ground.
        ground_points: Point cloud array (K, C) classified as ground.
        is_ground: Boolean mask of length N, True where points belong to ground.
        plane_model: Array [a, b, c, d] for plane ax + by + cz + d = 0, or None if not fit.
    """
    non_ground_points: np.ndarray
    ground_points: np.ndarray
    is_ground: np.ndarray
    plane_model: Optional[np.ndarray] = None


class RANSACGroundRemoval:
    """RANSAC ground plane segmentation and removal engine.

    Fits a dominant ground plane equation ax + by + cz + d = 0,
    validates the surface normal against the vertical axis (Z),
    and filters points within specified distance thresholds.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        ransac_threshold: float = 0.15,
        ransac_iterations: int = 100,
        ground_height_threshold: float = 0.3,
        max_normal_angle_deg: float = 30.0,
    ) -> None:
        """Initialize ground removal parameters.

        Args:
            config_path: Optional path to sensor_config.yaml.
            ransac_threshold: Max distance in meters to consider point on the plane.
            ransac_iterations: Number of RANSAC sampling iterations.
            ground_height_threshold: Height band (m) around plane considered ground.
            max_normal_angle_deg: Max angular deviation from vertical [0, 0, 1].
        """
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        self.ground_height_threshold = ground_height_threshold
        self.max_normal_angle_deg = max_normal_angle_deg

        if config_path is not None:
            self.load_config(config_path)

        # Min vertical component |nz| based on max deviation angle
        self.min_nz = np.cos(np.radians(self.max_normal_angle_deg))

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load ground removal parameters from YAML config file.

        Args:
            config_path: Path to sensor_config.yaml.
        """
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            logger.warning("Config path %s not found. Retaining default parameters.", cfg_path)
            return

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        prep_cfg = cfg.get("preprocessing", {}).get("ground_removal", {})
        if prep_cfg:
            self.ransac_threshold = float(prep_cfg.get("ransac_threshold", self.ransac_threshold))
            self.ransac_iterations = int(prep_cfg.get("ransac_iterations", self.ransac_iterations))
            self.ground_height_threshold = float(
                prep_cfg.get("ground_height_threshold", self.ground_height_threshold)
            )

    def _segment_plane_open3d(
        self, points_xyz: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Fit ground plane using Open3D CPU API.

        Args:
            points_xyz: (N, 3) point array.

        Returns:
            Tuple of (plane_model, inlier_indices).
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.ransac_threshold,
            ransac_n=3,
            num_iterations=self.ransac_iterations,
        )
        return np.asarray(plane_model, dtype=np.float32), np.asarray(inliers, dtype=np.int64)

    def _segment_plane_numpy(
        self, points_xyz: np.ndarray
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """NumPy fallback RANSAC plane segmentation when Open3D is unavailable.

        Args:
            points_xyz: (N, 3) point array.

        Returns:
            Tuple of (plane_model, inlier_indices).
        """
        n_points = points_xyz.shape[0]
        if n_points < 3:
            return None, np.array([], dtype=np.int64)

        best_inliers: np.ndarray = np.array([], dtype=np.int64)
        best_plane: Optional[np.ndarray] = None
        best_count = 0

        rng = np.random.default_rng(seed=42)

        for _ in range(self.ransac_iterations):
            sample_idx = rng.choice(n_points, size=3, replace=False)
            p1, p2, p3 = points_xyz[sample_idx]

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_mag = np.linalg.norm(normal)
            if norm_mag < 1e-7:
                continue
            normal /= norm_mag

            # Check if normal is sufficiently vertical
            if abs(normal[2]) < self.min_nz:
                continue

            d = -np.dot(normal, p1)
            # Distance from all points to plane: |ax + by + cz + d|
            distances = np.abs(np.dot(points_xyz, normal) + d)
            inliers = np.where(distances <= self.ransac_threshold)[0]

            if len(inliers) > best_count:
                best_count = len(inliers)
                best_inliers = inliers
                best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

        # Refine plane using least squares on inliers if found
        if best_plane is not None and len(best_inliers) >= 3:
            pts = points_xyz[best_inliers]
            centroid = pts.mean(axis=0)
            shifted = pts - centroid
            _, _, vh = np.linalg.svd(shifted, full_matrices=False)
            normal = vh[2]
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-7:
                normal /= norm_mag
                if normal[2] < 0:
                    normal = -normal
                d = -np.dot(normal, centroid)
                best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

        return best_plane, best_inliers

    def segment_ground(self, points: np.ndarray) -> GroundRemovalResult:
        """Segment input points into ground and non-ground subsets.

        Args:
            points: Array of shape (N, C) with C >= 3 (first 3 columns are x, y, z).

        Returns:
            GroundRemovalResult containing separated points and masks.
        """
        if points is None or len(points) == 0:
            empty = np.empty((0, points.shape[1] if points is not None and points.ndim > 1 else 3), dtype=np.float32)
            return GroundRemovalResult(
                non_ground_points=empty,
                ground_points=empty,
                is_ground=np.zeros(0, dtype=bool),
                plane_model=None,
            )

        pts_xyz = points[:, :3].astype(np.float64, copy=False)
        n_points = pts_xyz.shape[0]

        if n_points < 3:
            return GroundRemovalResult(
                non_ground_points=points.copy(),
                ground_points=points[:0].copy(),
                is_ground=np.zeros(n_points, dtype=bool),
                plane_model=None,
            )

        # Execute plane segmentation
        if HAS_OPEN3D:
            try:
                plane_model, inlier_indices = self._segment_plane_open3d(pts_xyz)
            except Exception as e:
                logger.warning("Open3D segment_plane failed (%s), falling back to NumPy RANSAC.", e)
                plane_model, inlier_indices = self._segment_plane_numpy(pts_xyz)
        else:
            plane_model, inlier_indices = self._segment_plane_numpy(pts_xyz)

        is_ground = np.zeros(n_points, dtype=bool)

        if plane_model is not None and len(inlier_indices) > 0:
            a, b, c, d = plane_model
            normal = np.array([a, b, c], dtype=np.float64)
            norm_mag = np.linalg.norm(normal)

            if norm_mag > 1e-7:
                normal /= norm_mag
                # Ensure normal points upwards (positive z)
                if normal[2] < 0:
                    normal = -normal
                    d = -d
                    plane_model = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

                # Validate vertical alignment
                if normal[2] >= self.min_nz:
                    # Compute signed distance to plane for all points
                    signed_dists = np.dot(pts_xyz, normal) + (d / norm_mag)
                    # Points within ground height threshold are ground
                    is_ground = (signed_dists >= -self.ground_height_threshold) & (
                        signed_dists <= self.ground_height_threshold
                    )
                else:
                    # Rejected: plane normal is not vertical (likely a wall/facade)
                    is_ground[inlier_indices] = False
                    plane_model = None

        non_ground_points = points[~is_ground]
        ground_points = points[is_ground]

        return GroundRemovalResult(
            non_ground_points=non_ground_points,
            ground_points=ground_points,
            is_ground=is_ground,
            plane_model=plane_model,
        )


def remove_ground(
    points: np.ndarray,
    config_path: Optional[Union[str, Path]] = None,
    ransac_threshold: float = 0.15,
    ransac_iterations: int = 100,
    ground_height_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience helper to remove ground points from a point cloud.

    Args:
        points: (N, C) numpy array of points.
        config_path: Optional path to YAML config.
        ransac_threshold: RANSAC distance threshold in meters.
        ransac_iterations: Iteration limit for RANSAC.
        ground_height_threshold: Thickness band in meters for ground.

    Returns:
        Tuple of (non_ground_points, ground_points).
    """
    engine = RANSACGroundRemoval(
        config_path=config_path,
        ransac_threshold=ransac_threshold,
        ransac_iterations=ransac_iterations,
        ground_height_threshold=ground_height_threshold,
    )
    res = engine.segment_ground(points)
    return res.non_ground_points, res.ground_points
