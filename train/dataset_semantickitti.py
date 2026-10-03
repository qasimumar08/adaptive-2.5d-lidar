"""SemanticKITTI PyTorch Dataset and DataLoader for spconv-triton Sparse U-Net.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
Provides:
  - SemanticKITTI bin/label loading and 28 -> 6 class remapping
  - Data augmentations (rotation, flip, scale, jitter, dropout)
  - Voxelization into spconv.SparseConvTensor compatible representation
  - CPU-first preprocessing with GPU transfer at the batch collate stage
  - Train/Val/Test sequence splitting per config/model_config.yaml
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from src.preprocessing.ground_removal import RANSACGroundRemoval
from src.preprocessing.voxelizer import Voxelizer, collate_sparse_voxels

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

    class Dataset:  # type: ignore
        """Minimal fallback Dataset base class when PyTorch is not installed."""
        def __len__(self) -> int:
            raise NotImplementedError
        def __getitem__(self, idx: int) -> Any:
            raise NotImplementedError

    class DataLoader:  # type: ignore
        """Minimal fallback DataLoader when PyTorch is not installed."""
        def __init__(self, dataset: Any, batch_size: int = 1, shuffle: bool = False, collate_fn: Any = None, **kwargs: Any) -> None:
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = collate_fn or (lambda x: x)

        def __iter__(self) -> Any:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                items = [self.dataset[idx] for idx in batch_indices]
                yield self.collate_fn(items)

        def __len__(self) -> int:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# Default label remap from config/model_config.yaml
DEFAULT_LABEL_REMAP: Dict[int, int] = {
    # 0: Drivable surface
    40: 0,  # road
    44: 0,  # parking
    48: 0,  # sidewalk
    49: 0,  # other-ground
    60: 0,  # lane-marking
    # 1: Non-drivable terrain
    72: 1,  # terrain
    # 2: Static obstacles
    50: 2,  # building
    51: 2,  # fence
    52: 2,  # other-structure
    70: 2,  # vegetation
    71: 2,  # trunk
    80: 2,  # pole
    81: 2,  # traffic-sign
    # 3: Dynamic vehicles
    10: 3,  # car
    11: 3,  # bicycle
    13: 3,  # bus
    15: 3,  # motorcycle
    16: 3,  # on-rails
    18: 3,  # truck
    20: 3,  # other-vehicle
    # 4: Dynamic pedestrians
    30: 4,  # person
    31: 4,  # bicyclist
    32: 4,  # motorcyclist
    # 5: Unknown / noise
    0: 5,   # unlabeled
    1: 5,   # outlier
}


def build_label_remap_lut(label_remap: Dict[int, int], default_class: int = 5) -> np.ndarray:
    """Build a fast 65536-entry NumPy LUT array for label remapping.

    Args:
        label_remap: Mapping of raw SemanticKITTI class ID to target class ID (0..5).
        default_class: Fallback class for unmapped raw IDs (class 5: unknown/noise).

    Returns:
        np.ndarray of shape (65536,) with mapped class IDs.
    """
    lut = np.full(65536, default_class, dtype=np.int64)
    for raw_id, mapped_id in label_remap.items():
        if 0 <= raw_id < 65536:
            lut[raw_id] = mapped_id
    return lut


class DataAugmentor:
    """Point cloud data augmentations configured via model_config.yaml."""

    def __init__(
        self,
        random_rotation: bool = True,
        rotation_range: Tuple[float, float] = (-np.pi, np.pi),
        random_flip: bool = True,
        random_scale: Tuple[float, float] = (0.9, 1.1),
        random_jitter: float = 0.02,
        random_drop: float = 0.1,
    ) -> None:
        """Initialize augmentation parameters.

        Args:
            random_rotation: Rotate cloud around Z axis.
            rotation_range: (min_rad, max_rad) for rotation angle.
            random_flip: Randomly flip X and Y coordinates.
            random_scale: (min_scale, max_scale) coordinate scaling.
            random_jitter: Std dev of Gaussian noise added to coordinates.
            random_drop: Fraction of points randomly dropped [0, 1).
        """
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range
        self.random_flip = random_flip
        self.random_scale = random_scale
        self.random_jitter = random_jitter
        self.random_drop = random_drop

    def augment(
        self, points: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply active augmentations to points and labels.

        Args:
            points: (N, C) array of points.
            labels: Optional (N,) array of integer class labels.

        Returns:
            Tuple of (augmented_points, augmented_labels).
        """
        if len(points) == 0:
            return points, labels

        pts = points.copy()
        lbls = labels.copy() if labels is not None else None

        # 1. Random rotation around Z axis
        if self.random_rotation:
            angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rot = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            pts[:, :3] = pts[:, :3] @ rot.T

        # 2. Random flip along X and Y axes
        if self.random_flip:
            if np.random.rand() > 0.5:
                pts[:, 0] = -pts[:, 0]
            if np.random.rand() > 0.5:
                pts[:, 1] = -pts[:, 1]

        # 3. Random scaling
        if self.random_scale is not None:
            scale = np.random.uniform(self.random_scale[0], self.random_scale[1])
            pts[:, :3] *= scale

        # 4. Random jitter
        if self.random_jitter > 0:
            noise = np.random.normal(0, self.random_jitter, size=pts[:, :3].shape).astype(np.float32)
            pts[:, :3] += noise

        # 5. Random point dropout
        if self.random_drop > 0 and len(pts) > 10:
            keep_prob = 1.0 - self.random_drop
            mask = np.random.rand(len(pts)) < keep_prob
            # Guard against dropping all points
            if np.any(mask):
                pts = pts[mask]
                if lbls is not None:
                    lbls = lbls[mask]

        return pts, lbls


class SemanticKITTIDataset(Dataset):
    """SemanticKITTI Dataset for variable-resolution 2.5D Lidar Perception."""

    def __init__(
        self,
        model_config_path: Union[str, Path] = "config/model_config.yaml",
        sensor_config_path: Union[str, Path] = "config/sensor_config.yaml",
        split: str = "train",  # 'train' | 'val' | 'test'
        data_root: Optional[Union[str, Path]] = None,
        sequences: Optional[Sequence[int]] = None,
        augment: Optional[bool] = None,
        apply_ground_removal: bool = False,
    ) -> None:
        """Initialize SemanticKITTI dataset.

        Args:
            model_config_path: Path to model_config.yaml.
            sensor_config_path: Path to sensor_config.yaml.
            split: 'train', 'val', or 'test'.
            data_root: Override root directory path if not from config.
            sequences: Explicit list of sequence integers to override split sequence defaults.
            augment: True to enable augmentation (defaults to True for 'train', False for val/test).
            apply_ground_removal: If True, executes RANSAC ground removal on input points.
        """
        self.split = split
        self.apply_ground_removal = apply_ground_removal

        # 1. Load Configurations
        self.model_cfg = self._load_yaml(model_config_path)
        self.sensor_cfg = self._load_yaml(sensor_config_path)

        # 2. Dataset Sequences and Paths
        ds_cfg = self.model_cfg.get("dataset", {})
        default_root = ds_cfg.get("root", "./data/semantickitti")
        self.data_root = Path(data_root if data_root is not None else default_root)

        if sequences is not None:
            self.sequence_ids = list(sequences)
        else:
            if split == "train":
                self.sequence_ids = ds_cfg.get("train_sequences", [0, 1, 2, 3, 4, 5, 6, 7, 9, 10])
            elif split == "val":
                self.sequence_ids = ds_cfg.get("val_sequences", [8])
            elif split == "test":
                self.sequence_ids = ds_cfg.get("test_sequences", list(range(11, 22)))
            else:
                self.sequence_ids = [0]

        # 3. Label Remapping
        remap_dict = ds_cfg.get("label_remap", DEFAULT_LABEL_REMAP)
        self.remap_lut = build_label_remap_lut(remap_dict, default_class=5)

        # 4. Augmentation Setup
        aug_cfg = self.model_cfg.get("training", {}).get("augmentation", {})
        self.augment = augment if augment is not None else (split == "train")
        self.augmentor = DataAugmentor(
            random_rotation=aug_cfg.get("random_rotation", True),
            rotation_range=tuple(aug_cfg.get("rotation_range", [-np.pi, np.pi])),
            random_flip=aug_cfg.get("random_flip", True),
            random_scale=tuple(aug_cfg.get("random_scale", [0.9, 1.1])),
            random_jitter=float(aug_cfg.get("random_jitter", 0.02)),
            random_drop=float(aug_cfg.get("random_drop", 0.1)),
        )

        # 5. Range filter from sensor config
        prep_cfg = self.sensor_cfg.get("preprocessing", {})
        range_cfg = prep_cfg.get("range_filter", {})
        self.min_range = float(range_cfg.get("min_range", 0.5))
        self.max_range = float(range_cfg.get("max_range", 100.0))

        # 6. Voxelizer
        unet_cfg = self.model_cfg.get("sparse_unet", {})
        self.voxelizer = Voxelizer(
            voxel_size=unet_cfg.get("voxel_size", [0.05, 0.05, 0.05]),
            point_cloud_range=unet_cfg.get("point_cloud_range", [-100, -100, -5, 100, 100, 5]),
            max_points_per_voxel=unet_cfg.get("max_points_per_voxel", 5),
            max_voxels=unet_cfg.get("max_voxels", 80000),
        )

        # 7. Optional Ground Removal Engine
        self.ground_engine = (
            RANSACGroundRemoval(
                ransac_threshold=prep_cfg.get("ground_removal", {}).get("ransac_threshold", 0.15),
                ransac_iterations=prep_cfg.get("ground_removal", {}).get("ransac_iterations", 100),
                ground_height_threshold=prep_cfg.get("ground_removal", {}).get("ground_height_threshold", 0.3),
            )
            if apply_ground_removal
            else None
        )

        # 8. Index scan files
        self.scan_files: List[Tuple[str, str]] = []
        self._index_files()

    def _load_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Safely load YAML config file or return empty dict if missing."""
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _index_files(self) -> None:
        """Scan data_root directory for existing sequence bin and label files."""
        self.scan_files = []
        for seq in self.sequence_ids:
            seq_str = f"{seq:02d}"
            # Check both paths: <root>/sequences/XX/velodyne and <root>/dataset/sequences/XX/velodyne
            seq_dir = self.data_root / "sequences" / seq_str
            if not seq_dir.exists():
                seq_dir = self.data_root / "dataset" / "sequences" / seq_str

            velo_dir = seq_dir / "velodyne"
            label_dir = seq_dir / "labels"

            if velo_dir.exists():
                bin_paths = sorted(glob.glob(str(velo_dir / "*.bin")))
                for b_path in bin_paths:
                    base_name = Path(b_path).stem
                    l_path = str(label_dir / f"{base_name}.label")
                    self.scan_files.append((b_path, l_path))

        if len(self.scan_files) == 0:
            logger.info("No SemanticKITTI files found at %s for sequences %s.", self.data_root, self.sequence_ids)

    def __len__(self) -> int:
        return len(self.scan_files)

    def load_scan(self, bin_path: str) -> np.ndarray:
        """Load SemanticKITTI .bin point cloud file.

        Format: float32 binary array of shape (N, 4) -> [x, y, z, remission].
        """
        scan = np.fromfile(bin_path, dtype=np.float32)
        return scan.reshape((-1, 4))

    def load_label(self, label_path: str) -> Optional[np.ndarray]:
        """Load and remap SemanticKITTI .label file.

        Format: uint32 binary array of shape (N,).
        Remaps raw class ID (lower 16 bits) to target 6 classes.
        """
        if not Path(label_path).exists():
            return None
        raw_labels = np.fromfile(label_path, dtype=np.uint32)
        raw_sem_ids = raw_labels & 0xFFFF
        return self.remap_lut[raw_sem_ids]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve and preprocess point cloud and labels for index idx.

        Returns:
            Dictionary containing:
                - 'points': (N, 4) array
                - 'labels': (N,) array or None
                - 'voxel_features': (M, 4) array
                - 'voxel_coords': (M, 3) [z, y, x] array
                - 'voxel_labels': (M,) array or None
                - 'point_to_voxel_idx': (N_valid,) mapping array
                - 'spatial_shape': [D, H, W]
                - 'bin_path': file path
        """
        bin_path, label_path = self.scan_files[idx]
        points = self.load_scan(bin_path)
        labels = self.load_label(label_path)

        return self.preprocess_cloud(points, labels, bin_path)

    def preprocess_cloud(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        bin_path: str = "",
    ) -> Dict[str, Any]:
        """Process points through range filtering, ground removal, augmentation, and voxelization.

        Args:
            points: (N, 4) point cloud array.
            labels: Optional (N,) mapped labels array.
            bin_path: Origin file identifier.

        Returns:
            Dictionary of processed features and sparse representation.
        """
        # 1. Range filtering
        r = np.linalg.norm(points[:, :3], axis=1)
        valid_range = (r >= self.min_range) & (r <= self.max_range)
        pts = points[valid_range]
        lbls = labels[valid_range] if labels is not None else None

        # 2. Optional Ground Removal
        if self.ground_engine is not None and len(pts) > 0:
            res = self.ground_engine.segment_ground(pts)
            pts = res.non_ground_points
            if lbls is not None:
                lbls = lbls[~res.is_ground]

        # 3. Data Augmentation (if active)
        if self.augment and len(pts) > 0:
            pts, lbls = self.augmentor.augment(pts, lbls)

        # 4. Voxelization
        v_res = self.voxelizer.voxelize(pts, lbls, deterministic=(not self.augment))

        return {
            "points": pts,
            "labels": lbls,
            "voxel_features": v_res.voxels,
            "voxel_coords": v_res.coordinates,
            "voxel_labels": v_res.voxel_labels,
            "point_to_voxel_idx": v_res.point_to_voxel_idx,
            "valid_point_mask": v_res.valid_point_mask,
            "spatial_shape": v_res.spatial_shape,
            "bin_path": bin_path,
        }


def create_semantickitti_dataloader(
    model_config_path: Union[str, Path] = "config/model_config.yaml",
    sensor_config_path: Union[str, Path] = "config/sensor_config.yaml",
    split: str = "train",
    data_root: Optional[Union[str, Path]] = None,
    sequences: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: int = 2,
    device: Optional[Union[str, Any]] = None,
) -> Tuple[SemanticKITTIDataset, Any]:
    """Create a configured SemanticKITTI Dataset and DataLoader.

    Args:
        model_config_path: Path to model_config.yaml.
        sensor_config_path: Path to sensor_config.yaml.
        split: 'train', 'val', or 'test'.
        data_root: Optional override of dataset root.
        sequences: Optional override of sequences list.
        batch_size: Batch size (defaults to batch_size from config/model_config.yaml).
        shuffle: Whether to shuffle (defaults to True for train, False for val/test).
        num_workers: DataLoader background worker processes.
        device: Device to transfer sparse tensors to in collate_fn (e.g. 'cuda').

    Returns:
        Tuple of (dataset, dataloader).
    """
    dataset = SemanticKITTIDataset(
        model_config_path=model_config_path,
        sensor_config_path=sensor_config_path,
        split=split,
        data_root=data_root,
        sequences=sequences,
    )

    if batch_size is None:
        batch_size = int(dataset.model_cfg.get("training", {}).get("batch_size", 4))

    if shuffle is None:
        shuffle = (split == "train")

    spatial_shape = dataset.voxelizer.spatial_shape

    def collate_fn(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_sparse_voxels(batch_items, spatial_shape=spatial_shape, device=device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if HAS_TORCH else 0,
        collate_fn=collate_fn,
    )

    return dataset, loader
