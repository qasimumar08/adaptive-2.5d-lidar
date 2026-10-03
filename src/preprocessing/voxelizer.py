"""Voxelization module for spconv-triton input format.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
Discretizes raw 3D point clouds into sparse 3D voxel grids suitable for
spconv-triton Sparse U-Net models on AMD ROCm.
Parameters are aligned with config/model_config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Try importing torch and spconv
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    spconv = None
    HAS_SPCONV = False


@dataclass
class SparseTensorCompat:
    """Lightweight compatible container mimicking spconv.SparseConvTensor.

    Ensures downstream code and tests function seamlessly in development/CPU
    environments where spconv-triton is not natively built.
    """
    features: Any
    indices: Any
    spatial_shape: List[int]
    batch_size: int = 1

    def replace_feature(self, new_features: Any) -> "SparseTensorCompat":
        """Return a copy with replaced features, matching spconv API."""
        return SparseTensorCompat(
            features=new_features,
            indices=self.indices,
            spatial_shape=self.spatial_shape,
            batch_size=self.batch_size,
        )

    def to(self, device: Any) -> "SparseTensorCompat":
        """Move underlying tensor features and indices to device."""
        if hasattr(self.features, "to"):
            new_features = self.features.to(device)
            new_indices = self.indices.to(device)
            return SparseTensorCompat(
                features=new_features,
                indices=new_indices,
                spatial_shape=self.spatial_shape,
                batch_size=self.batch_size,
            )
        return self

    def cuda(self) -> "SparseTensorCompat":
        """Move to CUDA/ROCm device."""
        return self.to("cuda")

    def cpu(self) -> "SparseTensorCompat":
        """Move to CPU device."""
        return self.to("cpu")

    def __repr__(self) -> str:
        f_shape = getattr(self.features, "shape", None)
        i_shape = getattr(self.indices, "shape", None)
        return (
            f"SparseTensorCompat(features={f_shape}, indices={i_shape}, "
            f"spatial_shape={self.spatial_shape}, batch_size={self.batch_size})"
        )


@dataclass
class VoxelizationResult:
    """Output container for voxelized point cloud representation.

    Attributes:
        voxels: Aggregated voxel features (M, C).
        coordinates: Voxel integer grid coordinates (M, 3) in [z_idx, y_idx, x_idx] order.
        num_points_per_voxel: Number of points aggregated per voxel (M,).
        point_to_voxel_idx: Mapping index from valid points to voxel row index (N_valid,).
        valid_point_mask: Boolean mask (N,) of points located inside point_cloud_range.
        voxel_labels: Majority class labels (M,) if point labels were supplied.
        spatial_shape: Grid spatial dimensions [D_z, H_y, W_x].
    """
    voxels: np.ndarray
    coordinates: np.ndarray  # [z, y, x]
    num_points_per_voxel: np.ndarray
    point_to_voxel_idx: np.ndarray
    valid_point_mask: np.ndarray
    voxel_labels: Optional[np.ndarray] = None
    spatial_shape: List[int] = None


class Voxelizer:
    """3D point cloud voxelizer producing sparse tensors for spconv-triton.

    Follows spconv convention:
      - Spatial shape: [D, H, W] -> [size_z, size_y, size_x]
      - Coordinate indices: [b, z, y, x] where b is batch index
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        voxel_size: Optional[List[float]] = None,
        point_cloud_range: Optional[List[float]] = None,
        max_points_per_voxel: int = 5,
        max_voxels: int = 80000,
    ) -> None:
        """Initialize voxelizer parameters.

        Args:
            config_path: Optional path to config/model_config.yaml.
            voxel_size: [vx, vy, vz] in meters (default [0.05, 0.05, 0.05]).
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] in meters.
            max_points_per_voxel: Maximum points aggregated per voxel.
            max_voxels: Maximum number of voxels permitted (sampling applied if exceeded).
        """
        # Default parameters from model_config.yaml
        self.voxel_size = np.array(voxel_size if voxel_size is not None else [0.05, 0.05, 0.05], dtype=np.float32)
        self.point_cloud_range = np.array(
            point_cloud_range if point_cloud_range is not None else [-100.0, -100.0, -5.0, 100.0, 100.0, 5.0],
            dtype=np.float32,
        )
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels

        if config_path is not None:
            self.load_config(config_path)

        self._compute_spatial_shape()

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load voxel parameters from YAML config file.

        Args:
            config_path: Path to model_config.yaml.
        """
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            logger.warning("Config path %s not found. Using defaults.", cfg_path)
            return

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        unet_cfg = cfg.get("sparse_unet", {})
        if unet_cfg:
            if "voxel_size" in unet_cfg:
                self.voxel_size = np.array(unet_cfg["voxel_size"], dtype=np.float32)
            if "point_cloud_range" in unet_cfg:
                self.point_cloud_range = np.array(unet_cfg["point_cloud_range"], dtype=np.float32)
            self.max_points_per_voxel = int(unet_cfg.get("max_points_per_voxel", self.max_points_per_voxel))
            self.max_voxels = int(unet_cfg.get("max_voxels", self.max_voxels))

        self._compute_spatial_shape()

    def _compute_spatial_shape(self) -> None:
        """Compute grid spatial shape [D_z, H_y, W_x] based on range and resolution."""
        # Range: [x_min, y_min, z_min, x_max, y_max, z_max]
        range_span = self.point_cloud_range[3:6] - self.point_cloud_range[0:3]
        grid_sizes = np.round(range_span / self.voxel_size).astype(np.int64)

        # spconv spatial_shape format is [D, H, W] -> [z, y, x]
        self.grid_size_x = int(grid_sizes[0])
        self.grid_size_y = int(grid_sizes[1])
        self.grid_size_z = int(grid_sizes[2])
        self.spatial_shape = [self.grid_size_z, self.grid_size_y, self.grid_size_x]

    def voxelize(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> VoxelizationResult:
        """Convert a point cloud into sparse voxels.

        Args:
            points: (N, C) numpy array, where first 3 columns are (x, y, z).
            labels: Optional (N,) numpy array of ground truth semantic class integers.
            deterministic: If True, preserves deterministic sampling when max_voxels is exceeded.

        Returns:
            VoxelizationResult containing voxel features, coordinates, and mappings.
        """
        if points is None or len(points) == 0:
            c_dim = points.shape[1] if (points is not None and points.ndim > 1) else 4
            return VoxelizationResult(
                voxels=np.empty((0, c_dim), dtype=np.float32),
                coordinates=np.empty((0, 3), dtype=np.int32),
                num_points_per_voxel=np.empty(0, dtype=np.int32),
                point_to_voxel_idx=np.empty(0, dtype=np.int64),
                valid_point_mask=np.zeros(0, dtype=bool),
                voxel_labels=np.empty(0, dtype=np.int64) if labels is not None else None,
                spatial_shape=self.spatial_shape,
            )

        xyz = points[:, :3].astype(np.float32, copy=False)
        n_points = points.shape[0]

        # 1. Filter points outside point_cloud_range
        in_range_mask = (
            (xyz[:, 0] >= self.point_cloud_range[0]) & (xyz[:, 0] < self.point_cloud_range[3]) &
            (xyz[:, 1] >= self.point_cloud_range[1]) & (xyz[:, 1] < self.point_cloud_range[4]) &
            (xyz[:, 2] >= self.point_cloud_range[2]) & (xyz[:, 2] < self.point_cloud_range[5])
        )

        valid_indices = np.where(in_range_mask)[0]
        if len(valid_indices) == 0:
            return VoxelizationResult(
                voxels=np.empty((0, points.shape[1]), dtype=np.float32),
                coordinates=np.empty((0, 3), dtype=np.int32),
                num_points_per_voxel=np.empty(0, dtype=np.int32),
                point_to_voxel_idx=np.empty(0, dtype=np.int64),
                valid_point_mask=in_range_mask,
                voxel_labels=np.empty(0, dtype=np.int64) if labels is not None else None,
                spatial_shape=self.spatial_shape,
            )

        filtered_points = points[valid_indices]
        filtered_xyz = xyz[valid_indices]
        filtered_labels = labels[valid_indices] if labels is not None else None

        # 2. Compute voxel grid coordinates
        # Coordinate calculation: floor((pos - min) / size)
        coord_x = np.floor((filtered_xyz[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]).astype(np.int32)
        coord_y = np.floor((filtered_xyz[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]).astype(np.int32)
        coord_z = np.floor((filtered_xyz[:, 2] - self.point_cloud_range[2]) / self.voxel_size[2]).astype(np.int32)

        # Clip coordinates within [0, grid_size - 1] to prevent edge boundary round-off
        coord_x = np.clip(coord_x, 0, self.grid_size_x - 1)
        coord_y = np.clip(coord_y, 0, self.grid_size_y - 1)
        coord_z = np.clip(coord_z, 0, self.grid_size_z - 1)

        # Stack into [z, y, x] per spconv convention
        coords_zyx = np.column_stack((coord_z, coord_y, coord_x))

        # 3. Find unique voxels and point mappings
        unique_coords, inverse_map, counts = np.unique(
            coords_zyx, axis=0, return_inverse=True, return_counts=True
        )
        num_voxels = unique_coords.shape[0]

        # 4. Limit to max_voxels if exceeded
        if num_voxels > self.max_voxels:
            if deterministic:
                selected_voxel_indices = np.arange(self.max_voxels, dtype=np.int64)
            else:
                rng = np.random.default_rng(seed=42)
                selected_voxel_indices = rng.choice(num_voxels, size=self.max_voxels, replace=False)
                selected_voxel_indices.sort()

            # Create remapped inverse map
            voxel_remap = np.full(num_voxels, -1, dtype=np.int64)
            voxel_remap[selected_voxel_indices] = np.arange(self.max_voxels)

            unique_coords = unique_coords[selected_voxel_indices]
            counts = counts[selected_voxel_indices]
            num_voxels = self.max_voxels

            # Update point-to-voxel mapping (-1 for points in dropped voxels)
            point_to_voxel = voxel_remap[inverse_map]
        else:
            point_to_voxel = inverse_map

        # 5. Aggregate features per voxel (mean pooling across points up to max_points_per_voxel)
        feature_dim = filtered_points.shape[1]
        voxel_features = np.zeros((num_voxels, feature_dim), dtype=np.float32)

        # Use bincount for vectorized summation of features for voxels
        valid_voxel_mask = (point_to_voxel >= 0)
        pts_in_valid_voxels = filtered_points[valid_voxel_mask]
        v_idx_of_pts = point_to_voxel[valid_voxel_mask]

        if len(v_idx_of_pts) > 0:
            for ch in range(feature_dim):
                summed = np.bincount(v_idx_of_pts, weights=pts_in_valid_voxels[:, ch], minlength=num_voxels)
                voxel_features[:, ch] = summed / np.maximum(counts, 1)

        # 6. Aggregate labels (majority vote) if provided
        voxel_labels: Optional[np.ndarray] = None
        if filtered_labels is not None:
            voxel_labels = np.zeros(num_voxels, dtype=np.int64)
            # Count class occurrences per voxel
            # Unique classes in filtered labels
            classes = np.unique(filtered_labels)
            class_counts = np.zeros((num_voxels, len(classes)), dtype=np.int32)
            class_to_idx = {c: i for i, c in enumerate(classes)}

            mapped_class_indices = np.array([class_to_idx[c] for c in filtered_labels[valid_voxel_mask]], dtype=np.int64)
            for ci, c_val in enumerate(classes):
                c_mask = (mapped_class_indices == ci)
                if np.any(c_mask):
                    b_counts = np.bincount(v_idx_of_pts[c_mask], minlength=num_voxels)
                    class_counts[:, ci] = b_counts

            majority_class_idx = np.argmax(class_counts, axis=1)
            voxel_labels = classes[majority_class_idx]

        return VoxelizationResult(
            voxels=voxel_features,
            coordinates=unique_coords.astype(np.int32),
            num_points_per_voxel=counts.astype(np.int32),
            point_to_voxel_idx=point_to_voxel,
            valid_point_mask=in_range_mask,
            voxel_labels=voxel_labels,
            spatial_shape=self.spatial_shape,
        )

    def to_sparse_tensor(
        self,
        voxels: Union[np.ndarray, "torch.Tensor"],
        coordinates: Union[np.ndarray, "torch.Tensor"],
        batch_size: int = 1,
        device: Optional[Union[str, Any]] = None,
    ) -> Any:
        """Construct a SparseConvTensor (or compatible representation).

        Args:
            voxels: Voxel features (M, C).
            coordinates: Voxel coordinates (M, 3) in [z, y, x] or (M, 4) with batch.
            batch_size: Batch dimension.
            device: Target device (e.g. 'cuda', 'cuda:0', 'cpu').

        Returns:
            spconv.SparseConvTensor or SparseTensorCompat.
        """
        # Ensure coordinates have batch index prepended: (M, 4) -> [b, z, y, x]
        if isinstance(coordinates, np.ndarray):
            if coordinates.ndim == 2 and coordinates.shape[1] == 3:
                b_zeros = np.zeros((coordinates.shape[0], 1), dtype=np.int32)
                indices = np.hstack((b_zeros, coordinates))
            else:
                indices = coordinates.astype(np.int32)
        else:
            if coordinates.ndim == 2 and coordinates.shape[1] == 3:
                b_zeros = torch.zeros((coordinates.shape[0], 1), dtype=torch.int32, device=coordinates.device)
                indices = torch.cat([b_zeros, coordinates.int()], dim=1)
            else:
                indices = coordinates.int()

        # Convert to torch.Tensor if torch is available
        if HAS_TORCH:
            if isinstance(voxels, np.ndarray):
                features_tensor = torch.from_numpy(voxels).float()
            else:
                features_tensor = voxels.float()

            if isinstance(indices, np.ndarray):
                indices_tensor = torch.from_numpy(indices).int()
            else:
                indices_tensor = indices.int()

            if device is not None:
                features_tensor = features_tensor.to(device)
                indices_tensor = indices_tensor.to(device)

            if HAS_SPCONV:
                return spconv.SparseConvTensor(
                    features=features_tensor,
                    indices=indices_tensor,
                    spatial_shape=self.spatial_shape,
                    batch_size=batch_size,
                )
            else:
                return SparseTensorCompat(
                    features=features_tensor,
                    indices=indices_tensor,
                    spatial_shape=self.spatial_shape,
                    batch_size=batch_size,
                )
        else:
            return SparseTensorCompat(
                features=voxels,
                indices=indices,
                spatial_shape=self.spatial_shape,
                batch_size=batch_size,
            )


def collate_sparse_voxels(
    batch_items: List[Dict[str, Any]],
    spatial_shape: List[int],
    device: Optional[Union[str, Any]] = None,
) -> Dict[str, Any]:
    """Collate function for PyTorch DataLoader to assemble batched sparse tensors.

    Args:
        batch_items: List of dictionary records produced by dataset.
        spatial_shape: Spatial grid shape [D_z, H_y, W_x].
        device: Device to place tensors on (e.g. 'cuda' for AMD GPU).

    Returns:
        Dictionary with 'sparse_tensor', 'voxel_labels', 'points', 'labels', etc.
    """
    batch_size = len(batch_items)
    voxels_list: List[np.ndarray] = []
    indices_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for b_idx, item in enumerate(batch_items):
        v = item["voxel_features"]
        coords = item["voxel_coords"]  # [z, y, x]
        num_v = coords.shape[0]

        # Prepend batch index: [b, z, y, x]
        b_col = np.full((num_v, 1), b_idx, dtype=np.int32)
        b_coords = np.hstack((b_col, coords.astype(np.int32)))

        voxels_list.append(v)
        indices_list.append(b_coords)
        if "voxel_labels" in item and item["voxel_labels"] is not None:
            labels_list.append(item["voxel_labels"])

    concat_voxels = np.vstack(voxels_list) if len(voxels_list) > 0 else np.empty((0, 4), dtype=np.float32)
    concat_indices = np.vstack(indices_list) if len(indices_list) > 0 else np.empty((0, 4), dtype=np.int32)

    concat_labels: Optional[np.ndarray] = None
    if len(labels_list) == batch_size:
        concat_labels = np.concatenate(labels_list)

    if HAS_TORCH:
        features_t = torch.from_numpy(concat_voxels).float()
        indices_t = torch.from_numpy(concat_indices).int()

        if device is not None:
            features_t = features_t.to(device)
            indices_t = indices_t.to(device)

        if HAS_SPCONV:
            sp_tensor = spconv.SparseConvTensor(
                features=features_t,
                indices=indices_t,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
            )
        else:
            sp_tensor = SparseTensorCompat(
                features=features_t,
                indices=indices_t,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
            )

        voxel_labels_t = torch.from_numpy(concat_labels).long() if concat_labels is not None else None
        if voxel_labels_t is not None and device is not None:
            voxel_labels_t = voxel_labels_t.to(device)
    else:
        sp_tensor = SparseTensorCompat(
            features=concat_voxels,
            indices=concat_indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
        voxel_labels_t = concat_labels

    return {
        "sparse_tensor": sp_tensor,
        "voxel_labels": voxel_labels_t,
        "raw_batch": batch_items,
    }
