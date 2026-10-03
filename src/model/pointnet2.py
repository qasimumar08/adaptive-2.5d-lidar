"""PointNet++ Fallback Semantic Segmentation Model using pure PyTorch operations.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
Implements PointNet++ Set Abstraction and Feature Propagation using pure PyTorch
tensor operations (no custom CUDA C++ extensions), guaranteeing out-of-the-box
execution on AMD ROCm GPUs and CPUs.
Configuration is loaded from config/model_config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Pure PyTorch Geometric Primitives (AMD ROCm Safe)
# =============================================================================

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate squared Euclidean distance between each pair of points.

    Args:
        src: Source points (B, N, C).
        dst: Target points (B, M, C).

    Returns:
        Pairwise squared distance matrix (B, N, M).
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points according to multi-dimensional indices.

    Args:
        points: Input point features (B, N, C).
        idx: Target point indices (B, S) or (B, S, K).

    Returns:
        Indexed points tensor of shape (B, S, C) or (B, S, K, C).
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Vectorized Farthest Point Sampling (FPS) using pure PyTorch.

    Args:
        xyz: Point coordinates (B, N, 3).
        npoint: Number of samples to collect.

    Returns:
        Indices of selected centroids (B, npoint).
    """
    device = xyz.device
    B, N, _ = xyz.shape
    npoint = min(npoint, N)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Ball query using pure PyTorch top-k.

    Args:
        radius: Local search radius.
        nsample: Maximum number of points per ball.
        xyz: All input points (B, N, 3).
        new_xyz: Query ball center coordinates (B, S, 3).

    Returns:
        Group indices (B, S, nsample).
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    nsample = min(nsample, N)
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)

    # Sort to get closest points
    group_idx = torch.topk(sqrdists, k=nsample, dim=-1, largest=False)[1]

    # Replace points outside radius with the closest point (group_idx[:, :, 0])
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = sqrdists.gather(dim=-1, index=group_idx) > (radius ** 2)
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample points with FPS and group neighbors via ball query.

    Args:
        npoint: Number of centroid points.
        radius: Ball query radius.
        nsample: Max points per group.
        xyz: Coordinates (B, N, 3).
        points: Features (B, N, D) or None.

    Returns:
        Tuple of:
          - new_xyz: Centroid coordinates (B, npoint, 3)
          - new_points: Normalized grouped coordinates + features (B, npoint, nsample, 3 + D)
    """
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
    # Center relative to centroid
    grouped_xyz_norm = grouped_xyz - new_xyz.view(new_xyz.shape[0], new_xyz.shape[1], 1, 3)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


# =============================================================================
# Set Abstraction & Feature Propagation Modules
# =============================================================================

class PointNetSetAbstraction(nn.Module):
    """Set Abstraction (SA) layer for PointNet++."""

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            xyz: Coordinates (B, 3, N).
            points: Features (B, D, N) or None.

        Returns:
            Tuple of (new_xyz: (B, 3, npoint), new_points: (B, mlp[-1], npoint)).
        """
        xyz = xyz.permute(0, 2, 1)  # (B, N, 3)
        if points is not None:
            points = points.permute(0, 2, 1)  # (B, N, D)

        new_xyz, new_points = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, points
        )
        # new_points shape: (B, npoint, nsample, 3 + D) -> transpose to (B, C, nsample, npoint)
        new_points = new_points.permute(0, 3, 2, 1)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        # Max pool across nsample dimension
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """Feature Propagation (FP) layer for PointNet++ with 3-NN interpolation."""

    def __init__(self, in_channel: int, mlp: List[int]) -> None:
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate points2 (at xyz2) to positions of xyz1 and concatenate points1.

        Args:
            xyz1: Finer points (B, 3, N).
            xyz2: Coarser points (B, 3, S).
            points1: Finer features (B, C1, N) or None.
            points2: Coarser features (B, C2, S).

        Returns:
            Propagated features (B, mlp[-1], N).
        """
        xyz1 = xyz1.permute(0, 2, 1)  # (B, N, 3)
        xyz2 = xyz2.permute(0, 2, 1)  # (B, S, 3)

        points2 = points2.permute(0, 2, 1)  # (B, S, C2)
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)  # (B, N, S)
            dists, idx = torch.sort(dists, dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3-NN

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points


# =============================================================================
# PointNet++ Semantic Segmentation Network
# =============================================================================

class PointNet2SemSeg(nn.Module):
    """PointNet++ Semantic Segmentation architecture using pure PyTorch.

    4 Set Abstraction stages + 4 Feature Propagation stages + Head.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 4,
        sa_configs: Optional[List[Dict[str, Any]]] = None,
        fp_mlps: Optional[List[List[int]]] = None,
    ) -> None:
        """Initialize PointNet++ layers.

        Args:
            num_classes: Output semantic classes (default: 6).
            in_channels: Feature channels (x, y, z, + extra features, default 4).
            sa_configs: List of 4 dicts with keys (npoint, radius, nsample, mlp).
            fp_mlps: List of 4 mlp channel lists for feature propagation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        extra_dim = in_channels - 3

        # Default SA layers from config/model_config.yaml
        if sa_configs is None:
            sa_configs = [
                {"npoint": 4096, "radius": 0.2, "nsample": 32, "mlp": [32, 32, 64]},
                {"npoint": 1024, "radius": 0.4, "nsample": 64, "mlp": [64, 64, 128]},
                {"npoint": 256,  "radius": 0.8, "nsample": 128, "mlp": [128, 128, 256]},
                {"npoint": 64,   "radius": 1.6, "nsample": 256, "mlp": [256, 256, 512]},
            ]

        # 4 Set Abstraction Layers
        sa1_out = sa_configs[0]["mlp"][-1]
        sa2_out = sa_configs[1]["mlp"][-1]
        sa3_out = sa_configs[2]["mlp"][-1]
        sa4_out = sa_configs[3]["mlp"][-1]

        self.sa1 = PointNetSetAbstraction(
            sa_configs[0]["npoint"], sa_configs[0]["radius"], sa_configs[0]["nsample"],
            3 + extra_dim, sa_configs[0]["mlp"]
        )
        self.sa2 = PointNetSetAbstraction(
            sa_configs[1]["npoint"], sa_configs[1]["radius"], sa_configs[1]["nsample"],
            3 + sa1_out, sa_configs[1]["mlp"]
        )
        self.sa3 = PointNetSetAbstraction(
            sa_configs[2]["npoint"], sa_configs[2]["radius"], sa_configs[2]["nsample"],
            3 + sa2_out, sa_configs[2]["mlp"]
        )
        self.sa4 = PointNetSetAbstraction(
            sa_configs[3]["npoint"], sa_configs[3]["radius"], sa_configs[3]["nsample"],
            3 + sa3_out, sa_configs[3]["mlp"]
        )

        # 4 Feature Propagation Layers
        self.fp4 = PointNetFeaturePropagation(sa4_out + sa3_out, [sa3_out, sa3_out])
        self.fp3 = PointNetFeaturePropagation(sa3_out + sa2_out, [sa3_out, sa2_out])
        self.fp2 = PointNetFeaturePropagation(sa2_out + sa1_out, [sa2_out, sa2_out])
        self.fp1 = PointNetFeaturePropagation(sa2_out + extra_dim, [128, 128])

        # Head classification
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "config/model_config.yaml") -> "PointNet2SemSeg":
        """Factory method to construct PointNet2SemSeg from YAML config file.

        Args:
            config_path: Path to model_config.yaml.

        Returns:
            Initialized PointNet2SemSeg model instance.
        """
        p = Path(config_path)
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        pn_cfg = cfg.get("pointnet2", {})
        num_classes = int(pn_cfg.get("num_classes", 6))
        sa_layers = pn_cfg.get("sa_layers", None)
        return cls(num_classes=num_classes, in_channels=4, sa_configs=sa_layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            points: Input tensor (B, C, N) or (B, N, C) where C >= 3 (first 3 are x, y, z).

        Returns:
            Logits tensor of shape (B, num_classes, N).
        """
        # Ensure points is (B, C, N)
        if points.shape[1] != self.in_channels and points.shape[2] == self.in_channels:
            points = points.permute(0, 2, 1)

        xyz = points[:, :3, :]
        features = points[:, 3:, :] if points.shape[1] > 3 else None

        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature Propagation
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, features, l1_points)

        # Classifier
        logits = self.classifier(l0_points)  # (B, num_classes, N)
        return logits
