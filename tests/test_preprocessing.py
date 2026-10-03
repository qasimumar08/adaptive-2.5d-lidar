"""Unit tests for point cloud preprocessing and SemanticKITTI dataloader.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
Tests:
  - RANSAC ground plane removal (with both Open3D and NumPy fallback)
  - Voxelizer (spatial shape, coordinate order [z, y, x], feature & label aggregation)
  - Ego-motion compensator (timestamps, linear/angular deskewing)
  - SemanticKITTI dataset (28 -> 6 label remapping, augmentations, batch collation)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.ground_removal import (
    GroundRemovalResult,
    RANSACGroundRemoval,
    remove_ground,
)
from src.preprocessing.voxelizer import (
    SparseTensorCompat,
    VoxelizationResult,
    Voxelizer,
    collate_sparse_voxels,
)
from src.preprocessing.ego_motion import EgoMotionCompensator
from train.dataset_semantickitti import (
    DataAugmentor,
    SemanticKITTIDataset,
    build_label_remap_lut,
    DEFAULT_LABEL_REMAP,
)


# =============================================================================
# 1. Ground Removal Tests
# =============================================================================

class TestGroundRemoval:
    """Tests for RANSAC ground plane segmentation."""

    def test_synthetic_ground_separation(self) -> None:
        """Verify ground points are separated from elevated obstacle points."""
        rng = np.random.default_rng(42)

        # 500 ground points on plane z = -1.73 with small noise (+/- 0.05m)
        x_g = rng.uniform(-20.0, 20.0, 500)
        y_g = rng.uniform(-20.0, 20.0, 500)
        z_g = -1.73 + rng.normal(0, 0.03, 500)
        intensity_g = rng.uniform(0.1, 0.5, 500)
        ground_pts = np.column_stack((x_g, y_g, z_g, intensity_g)).astype(np.float32)

        # 200 obstacle points above ground (z between 0.0 and 2.5m)
        x_o = rng.uniform(-10.0, 10.0, 200)
        y_o = rng.uniform(5.0, 15.0, 200)
        z_o = rng.uniform(0.0, 2.5, 200)
        intensity_o = rng.uniform(0.5, 1.0, 200)
        obstacle_pts = np.column_stack((x_o, y_o, z_o, intensity_o)).astype(np.float32)

        points = np.vstack((ground_pts, obstacle_pts))

        remover = RANSACGroundRemoval(
            ransac_threshold=0.15,
            ransac_iterations=100,
            ground_height_threshold=0.3,
            max_normal_angle_deg=25.0,
        )
        res = remover.segment_ground(points)

        assert isinstance(res, GroundRemovalResult)
        assert len(res.ground_points) > 400, "Most ground points should be identified"
        assert len(res.non_ground_points) >= 190, "Obstacle points should not be removed"
        assert res.plane_model is not None

        # Verify normal points upwards along Z
        normal = res.plane_model[:3]
        assert abs(normal[2]) > 0.85, f"Normal Z component should be near 1.0, got {normal[2]}"

    def test_open3d_and_numpy_segmentation_direct(self) -> None:
        """Verify both Open3D and NumPy fallback plane segmentation directly."""
        rng = np.random.default_rng(42)
        x_g = rng.uniform(-10.0, 10.0, 100)
        y_g = rng.uniform(-10.0, 10.0, 100)
        z_g = -1.5 + rng.normal(0, 0.02, 100)
        ground_pts = np.column_stack((x_g, y_g, z_g)).astype(np.float64)

        remover = RANSACGroundRemoval(ransac_threshold=0.1, ransac_iterations=50)

        # Test NumPy fallback method directly
        plane_np, inliers_np = remover._segment_plane_numpy(ground_pts)
        assert plane_np is not None
        assert len(inliers_np) >= 80

        # Test Open3D method if installed
        from src.preprocessing.ground_removal import HAS_OPEN3D
        if HAS_OPEN3D:
            plane_o3d, inliers_o3d = remover._segment_plane_open3d(ground_pts)
            assert plane_o3d is not None
            assert len(inliers_o3d) >= 80

    def test_empty_and_small_clouds(self) -> None:
        """Verify graceful handling of empty or degenerate point clouds."""
        remover = RANSACGroundRemoval()

        # Empty cloud
        empty_pts = np.empty((0, 4), dtype=np.float32)
        res_empty = remover.segment_ground(empty_pts)
        assert len(res_empty.ground_points) == 0
        assert len(res_empty.non_ground_points) == 0
        assert res_empty.plane_model is None

        # Degenerate cloud (< 3 points)
        two_pts = np.array([[0.0, 0.0, 0.0, 0.1], [1.0, 1.0, 0.0, 0.2]], dtype=np.float32)
        res_small = remover.segment_ground(two_pts)
        assert len(res_small.non_ground_points) == 2
        assert len(res_small.ground_points) == 0

    def test_convenience_function(self) -> None:
        """Verify remove_ground functional wrapper."""
        pts = np.array([
            [0.0, 0.0, -1.5, 0.1],
            [1.0, 0.0, -1.5, 0.1],
            [0.0, 1.0, -1.5, 0.1],
            [5.0, 5.0, 1.0, 0.9],
        ], dtype=np.float32)

        non_ground, ground = remove_ground(pts, ransac_threshold=0.1)
        assert len(ground) >= 3
        assert len(non_ground) >= 1

    def test_config_loading(self) -> None:
        """Verify parameters load properly from config/sensor_config.yaml."""
        config_file = "config/sensor_config.yaml"
        if Path(config_file).exists():
            remover = RANSACGroundRemoval(config_path=config_file)
            assert remover.ransac_threshold == 0.15
            assert remover.ransac_iterations == 100
            assert remover.ground_height_threshold == 0.3


# =============================================================================
# 2. Voxelizer Tests
# =============================================================================

class TestVoxelizer:
    """Tests for 3D point cloud voxelization."""

    def test_spatial_shape_from_config(self) -> None:
        """Verify spatial shape [D_z, H_y, W_x] corresponds to config specifications."""
        voxelizer = Voxelizer(
            voxel_size=[0.05, 0.05, 0.05],
            point_cloud_range=[-100.0, -100.0, -5.0, 100.0, 100.0, 5.0],
        )
        # Size Z = (5 - (-5)) / 0.05 = 200
        # Size Y = (100 - (-100)) / 0.05 = 4000
        # Size X = (100 - (-100)) / 0.05 = 4000
        assert voxelizer.spatial_shape == [200, 4000, 4000]

    def test_voxelize_simple_points(self) -> None:
        """Verify points inside same voxel aggregate into 1 voxel."""
        voxelizer = Voxelizer(
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-10.0, -10.0, -2.0, 10.0, 10.0, 2.0],
        )

        # 3 points falling in the same voxel near origin
        pts = np.array([
            [0.01, 0.02, 0.03, 1.0],
            [0.02, 0.03, 0.04, 2.0],
            [0.03, 0.04, 0.05, 3.0],
            # 1 point in a different voxel
            [5.0, 5.0, 0.0, 10.0],
        ], dtype=np.float32)

        labels = np.array([0, 0, 1, 3], dtype=np.int64)

        res = voxelizer.voxelize(pts, labels=labels)
        assert isinstance(res, VoxelizationResult)
        assert len(res.voxels) == 2, "Should produce exactly 2 unique voxels"

        # Coordinates must follow [z, y, x] format
        assert res.coordinates.shape == (2, 3)

        # First voxel feature should be mean of the 3 points
        expected_feature_0 = np.mean(pts[:3], axis=0)
        assert np.allclose(res.voxels[0], expected_feature_0, atol=1e-5)

        # Majority label for voxel 0 should be 0 (two votes for 0, one vote for 1)
        assert res.voxel_labels[0] == 0
        assert res.voxel_labels[1] == 3

    def test_range_filtering(self) -> None:
        """Verify out-of-range points are filtered out."""
        voxelizer = Voxelizer(
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-10.0, -10.0, -2.0, 10.0, 10.0, 2.0],
        )

        pts = np.array([
            [0.0, 0.0, 0.0, 1.0],     # inside
            [15.0, 0.0, 0.0, 2.0],    # outside X max
            [0.0, -15.0, 0.0, 3.0],   # outside Y min
            [0.0, 0.0, 5.0, 4.0],     # outside Z max
        ], dtype=np.float32)

        res = voxelizer.voxelize(pts)
        assert len(res.voxels) == 1
        assert np.sum(res.valid_point_mask) == 1

    def test_to_sparse_tensor(self) -> None:
        """Verify to_sparse_tensor produces compatible sparse representation."""
        voxelizer = Voxelizer(
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-5.0, -5.0, -1.0, 5.0, 5.0, 1.0],
        )
        pts = np.array([[1.0, 2.0, 0.0, 0.5], [2.0, 3.0, 0.0, 0.8]], dtype=np.float32)
        res = voxelizer.voxelize(pts)

        sp_tensor = voxelizer.to_sparse_tensor(res.voxels, res.coordinates, batch_size=1)
        assert hasattr(sp_tensor, "features")
        assert hasattr(sp_tensor, "indices")
        assert hasattr(sp_tensor, "spatial_shape")
        # Indices must have shape (M, 4) with batch index prepended
        assert sp_tensor.indices.shape[1] == 4

    def test_collate_sparse_voxels(self) -> None:
        """Verify collate_sparse_voxels aggregates multiple samples with batch indices."""
        spatial_shape = [20, 100, 100]

        sample_0 = {
            "voxel_features": np.ones((5, 4), dtype=np.float32),
            "voxel_coords": np.zeros((5, 3), dtype=np.int32),
            "voxel_labels": np.zeros(5, dtype=np.int64),
        }
        sample_1 = {
            "voxel_features": np.ones((7, 4), dtype=np.float32) * 2.0,
            "voxel_coords": np.ones((7, 3), dtype=np.int32),
            "voxel_labels": np.ones(7, dtype=np.int64),
        }

        batched = collate_sparse_voxels([sample_0, sample_1], spatial_shape=spatial_shape)
        sp_tensor = batched["sparse_tensor"]

        assert sp_tensor.batch_size == 2
        assert len(sp_tensor.features) == 12  # 5 + 7
        assert len(sp_tensor.indices) == 12

        # Verify batch index [b, z, y, x]
        indices = sp_tensor.indices
        if hasattr(indices, "numpy"):
            indices = indices.numpy()

        assert np.all(indices[:5, 0] == 0)
        assert np.all(indices[5:, 0] == 1)


# =============================================================================
# 3. Ego-Motion Compensation Tests
# =============================================================================

class TestEgoMotionCompensator:
    """Tests for scan deskewing."""

    def test_passthrough_when_disabled(self) -> None:
        """Verify points returned unchanged when compensator is disabled."""
        compensator = EgoMotionCompensator(enabled=False)
        pts = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
        res = compensator.compensate(pts, timestamps=np.array([0.05]))
        assert np.allclose(pts, res)

    def test_linear_deskewing(self) -> None:
        """Verify translational deskewing adjusts coordinates proportional to velocity * time."""
        compensator = EgoMotionCompensator(enabled=True)

        # Vehicle moving forward along X at 10 m/s
        pts = np.array([
            [10.0, 0.0, 0.0, 0.5],  # at start of scan t=0.0s (dt = -0.1s from end)
            [10.0, 0.0, 0.0, 0.5],  # at end of scan t=0.1s (dt = 0.0s from end)
        ], dtype=np.float32)
        timestamps = np.array([0.0, 0.1], dtype=np.float32)
        linear_vel = np.array([10.0, 0.0, 0.0], dtype=np.float32)

        deskewed = compensator.compensate(pts, timestamps=timestamps, linear_velocity=linear_vel)

        # At t=0.1 (end of scan), no translation correction
        assert np.isclose(deskewed[1, 0], 10.0)
        # At t=0.0 (t_rel = -0.1s), delta_x = 10 * (-0.1) = -1.0 -> x = 10.0 - 1.0 = 9.0
        assert np.isclose(deskewed[0, 0], 9.0, atol=1e-4)

    def test_estimate_timestamps_from_azimuth(self) -> None:
        """Verify azimuth-based acquisition time estimation for spinning Lidars."""
        compensator = EgoMotionCompensator()
        # Points along +X, +Y, -X, -Y
        pts = np.array([
            [10.0, 0.0, 0.0],   # angle 0 -> t = 0
            [0.0, 10.0, 0.0],   # angle pi/2 -> t = 0.025s (for 0.1s period)
            [-10.0, 0.0, 0.0],  # angle pi -> t = 0.05s
            [0.0, -10.0, 0.0],  # angle 3pi/2 -> t = 0.075s
        ], dtype=np.float32)

        t = compensator.estimate_timestamps_from_azimuth(pts, scan_period=0.1)
        assert len(t) == 4
        assert np.isclose(t[0], 0.0, atol=1e-3)
        assert np.isclose(t[1], 0.025, atol=1e-3)
        assert np.isclose(t[2], 0.05, atol=1e-3)
        assert np.isclose(t[3], 0.075, atol=1e-3)


# =============================================================================
# 4. SemanticKITTI Dataloader Tests
# =============================================================================

class TestSemanticKITTIDataset:
    """Tests for SemanticKITTI label remapping and dataset pipeline."""

    def test_label_remapping_lut(self) -> None:
        """Verify remapping of all 28 classes into the target 6 classes."""
        lut = build_label_remap_lut(DEFAULT_LABEL_REMAP, default_class=5)

        # Drivable surface (0)
        assert lut[40] == 0  # road
        assert lut[44] == 0  # parking
        assert lut[48] == 0  # sidewalk
        assert lut[49] == 0  # other-ground
        assert lut[60] == 0  # lane-marking

        # Non-drivable terrain (1)
        assert lut[72] == 1  # terrain

        # Static obstacles (2)
        assert lut[50] == 2  # building
        assert lut[51] == 2  # fence
        assert lut[70] == 2  # vegetation
        assert lut[80] == 2  # pole

        # Dynamic vehicles (3)
        assert lut[10] == 3  # car
        assert lut[11] == 3  # bicycle
        assert lut[18] == 3  # truck

        # Dynamic pedestrians (4)
        assert lut[30] == 4  # person
        assert lut[31] == 4  # bicyclist

        # Unknown / noise (5)
        assert lut[0] == 5   # unlabeled
        assert lut[1] == 5   # outlier
        assert lut[999] == 5 # unmapped raw ID

    def test_data_augmentation(self) -> None:
        """Verify data augmentations modify point clouds within defined bounds."""
        augmentor = DataAugmentor(
            random_rotation=True,
            rotation_range=(-np.pi, np.pi),
            random_flip=False,
            random_scale=(0.95, 1.05),
            random_jitter=0.01,
            random_drop=0.2,
        )

        pts = np.random.uniform(-10.0, 10.0, size=(100, 4)).astype(np.float32)
        lbls = np.random.randint(0, 6, size=(100,)).astype(np.int64)

        aug_pts, aug_lbls = augmentor.augment(pts, lbls)

        # Points dropped according to drop rate ~20%
        assert len(aug_pts) < 100
        assert len(aug_pts) == len(aug_lbls)
        # Coordinate shape preserved
        assert aug_pts.shape[1] == 4

    def test_dataset_pipeline_with_mock_files(self) -> None:
        """Verify end-to-end dataset indexing, loading, and voxelization using synthetic sequence."""
        temp_dir = tempfile.mkdtemp()
        try:
            seq_dir = Path(temp_dir) / "sequences" / "00"
            velo_dir = seq_dir / "velodyne"
            label_dir = seq_dir / "labels"
            velo_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            # Create mock scan (100 points)
            pts = np.random.uniform(-20.0, 20.0, size=(100, 4)).astype(np.float32)
            pts[:, 2] = np.random.uniform(-1.5, 2.0, size=100)  # Z range
            bin_file = velo_dir / "000000.bin"
            pts.tofile(str(bin_file))

            # Create mock labels (uint32)
            # Mix road (40), car (10), terrain (72)
            raw_labels = np.random.choice([40, 10, 72], size=100).astype(np.uint32)
            label_file = label_dir / "000000.label"
            raw_labels.tofile(str(label_file))

            # Instantiate dataset pointing to temporary directory
            dataset = SemanticKITTIDataset(
                model_config_path="config/model_config.yaml",
                sensor_config_path="config/sensor_config.yaml",
                data_root=temp_dir,
                sequences=[0],
                augment=False,
            )

            assert len(dataset) == 1

            sample = dataset[0]
            assert "points" in sample
            assert "labels" in sample
            assert "voxel_features" in sample
            assert "voxel_coords" in sample
            assert "voxel_labels" in sample

            # Verify labels are remapped to 0..5
            assert np.all(sample["labels"] >= 0) and np.all(sample["labels"] < 6)
            assert np.all(sample["voxel_labels"] >= 0) and np.all(sample["voxel_labels"] < 6)

            # Test batch collation
            batched = collate_sparse_voxels([sample], spatial_shape=sample["spatial_shape"])
            assert "sparse_tensor" in batched
            assert batched["sparse_tensor"].batch_size == 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
