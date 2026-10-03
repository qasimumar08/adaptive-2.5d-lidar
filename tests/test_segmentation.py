"""Smoke tests and unit tests for Semantic Segmentation models, losses, and evaluation.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
Tests:
  - Sparse U-Net forward pass & gradient flow with random sparse tensor
  - PointNet++ pure PyTorch forward pass & gradient flow
  - Weighted Cross-Entropy + Lovász-Softmax loss
  - Confusion matrix and mIoU metric calculations
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model.sparse_unet import SparseUNet, HAS_SPCONV
from src.model.pointnet2 import PointNet2SemSeg, farthest_point_sample, query_ball_point
from src.model.losses import CombinedLoss, LovaszSoftmaxLoss, lovasz_softmax
from src.preprocessing.voxelizer import Voxelizer
from train.evaluate import compute_metrics_from_confusion_matrix, format_metrics_table


# =============================================================================
# 1. Sparse U-Net Tests
# =============================================================================

class TestSparseUNet:
    """Smoke tests for Sparse U-Net with spconv-triton representation."""

    def test_sparse_unet_from_config(self) -> None:
        """Verify model loads architecture parameters from config/model_config.yaml."""
        model = SparseUNet.from_config("config/model_config.yaml")
        assert model.in_channels == 4
        assert model.num_classes == 6
        assert model.encoder_channels == [16, 32, 64, 128, 256]
        assert model.decoder_channels == [256, 128, 64, 32, 16]

    def test_sparse_unet_forward_and_backward(self) -> None:
        """Smoke test forward pass and gradient backpropagation with random sparse tensor."""
        model = SparseUNet.from_config("config/model_config.yaml")
        model.train()

        num_voxels = 120
        spatial_shape = [200, 4000, 4000]

        voxelizer = Voxelizer(
            voxel_size=[0.05, 0.05, 0.05],
            point_cloud_range=[-100.0, -100.0, -5.0, 100.0, 100.0, 5.0],
        )

        # Generate synthetic voxel features (num_voxels, 4)
        features = np.random.randn(num_voxels, 4).astype(np.float32)
        # Coordinates in [z, y, x]
        coords_z = np.random.randint(0, 200, size=num_voxels)
        coords_y = np.random.randint(0, 4000, size=num_voxels)
        coords_x = np.random.randint(0, 4000, size=num_voxels)
        coords = np.column_stack((coords_z, coords_y, coords_x)).astype(np.int32)

        sp_tensor = voxelizer.to_sparse_tensor(features, coords, batch_size=1)

        # Forward pass
        logits = model(sp_tensor)
        assert logits.shape == (num_voxels, 6), f"Expected shape ({num_voxels}, 6), got {logits.shape}"

        # Backward pass with loss
        dummy_targets = torch.randint(0, 6, (num_voxels,), dtype=torch.long)
        criterion = CombinedLoss(config_path="config/model_config.yaml")
        loss_dict = criterion(logits, dummy_targets)

        loss = loss_dict["loss"]
        assert torch.isfinite(loss), "Loss must be finite"
        assert loss.item() > 0.0, "Loss must be strictly positive"

        loss.backward()

        # Check gradients exist
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grads, "Model parameters should have valid non-zero gradients"


# =============================================================================
# 2. PointNet++ Tests
# =============================================================================

class TestPointNet2:
    """Smoke tests for pure PyTorch PointNet++ fallback model."""

    def test_pointnet2_geometric_primitives(self) -> None:
        """Verify farthest point sampling and ball query."""
        B, N = 2, 64
        xyz = torch.randn(B, N, 3)

        # Test FPS
        npoint = 16
        centroids = farthest_point_sample(xyz, npoint)
        assert centroids.shape == (B, npoint)
        assert centroids.min() >= 0 and centroids.max() < N

        # Test ball query
        new_xyz = xyz[:, :npoint, :]
        idx = query_ball_point(radius=1.0, nsample=8, xyz=xyz, new_xyz=new_xyz)
        assert idx.shape == (B, npoint, 8)

    def test_pointnet2_forward_backward(self) -> None:
        """Verify PointNet++ forward pass and backward pass on random point cloud."""
        # Lightweight test configuration
        sa_configs = [
            {"npoint": 64, "radius": 0.4, "nsample": 16, "mlp": [16, 16, 32]},
            {"npoint": 32, "radius": 0.8, "nsample": 16, "mlp": [32, 32, 64]},
            {"npoint": 16, "radius": 1.2, "nsample": 16, "mlp": [64, 64, 128]},
            {"npoint": 8,  "radius": 1.6, "nsample": 8,  "mlp": [128, 128, 256]},
        ]
        model = PointNet2SemSeg(num_classes=6, in_channels=4, sa_configs=sa_configs)
        model.train()

        B, N = 2, 128
        points = torch.randn(B, 4, N)
        logits = model(points)

        assert logits.shape == (B, 6, N), f"Expected shape ({B}, 6, {N}), got {logits.shape}"

        targets = torch.randint(0, 6, (B, N), dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        loss.backward()

        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grads, "Gradients should flow to PointNet++ parameters"


# =============================================================================
# 3. Loss Functions Tests
# =============================================================================

class TestLosses:
    """Tests for Weighted Cross-Entropy and Lovász-Softmax losses."""

    def test_lovasz_softmax(self) -> None:
        """Verify Lovász-Softmax loss on random probabilities."""
        probas = torch.softmax(torch.randn(50, 6), dim=-1)
        labels = torch.randint(0, 6, (50,))

        loss = lovasz_softmax(probas, labels, classes="present")
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_combined_loss_config_and_grad(self) -> None:
        """Verify CombinedLoss loads config weights and propagates gradients."""
        loss_fn = CombinedLoss(config_path="config/model_config.yaml", num_classes=6)

        logits = torch.randn(64, 6, requires_grad=True)
        targets = torch.randint(0, 6, (64,), dtype=torch.long)

        out = loss_fn(logits, targets)
        assert "loss" in out
        assert "ce_loss" in out
        assert "lovasz_loss" in out

        total_loss = out["loss"]
        assert torch.isfinite(total_loss)
        assert total_loss.item() > 0.0

        total_loss.backward()
        assert logits.grad is not None and logits.grad.abs().sum() > 0


# =============================================================================
# 4. Evaluation Metrics Tests
# =============================================================================

class TestEvaluationMetrics:
    """Tests for confusion matrix metrics and table formatting."""

    def test_compute_metrics(self) -> None:
        """Verify metric calculation on synthetic confusion matrix."""
        # 3 classes:
        # Class 0: 80 TP, 10 predicted as 1, 10 predicted as 2
        # Class 1: 5 predicted as 0, 70 TP, 25 predicted as 2
        # Class 2: 5 predicted as 0, 5 predicted as 1, 90 TP
        cm = np.array([
            [80, 10, 10],
            [5,  70, 25],
            [5,  5,  90],
        ], dtype=np.int64)

        metrics = compute_metrics_from_confusion_matrix(cm)
        assert "mIoU" in metrics
        assert "overall_accuracy" in metrics
        assert "per_class" in metrics

        # Class 0: TP=80, FP=10, FN=20 -> IoU = 80 / (80 + 10 + 20) = 80 / 110 = 0.7272
        iou_0 = metrics["per_class"]["drivable_surface"]["iou"]
        assert np.isclose(iou_0, 80.0 / 110.0, atol=1e-3)

        assert 0.0 <= metrics["mIoU"] <= 1.0
        assert 0.0 <= metrics["overall_accuracy"] <= 1.0

        table_str = format_metrics_table(metrics)
        assert "Mean IoU (mIoU)" in table_str
        assert "drivable_surface" in table_str
