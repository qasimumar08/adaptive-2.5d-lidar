"""Loss functions for Lidar semantic segmentation.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
Implements:
  - Weighted Cross-Entropy Loss with class weights from config
  - Lovász-Softmax loss (vendored inline, optimizing Jaccard / mIoU surrogate)
  - CombinedLoss (Cross-Entropy + Lovász-Softmax blend)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import yaml

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object  # type: ignore
    F = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# =============================================================================
# Lovász-Softmax Core (Berman et al., CVPR 2018)
# Vendored inline to eliminate external pip dependency issues on ROCm.
# =============================================================================

def lovasz_grad(gt_sorted: "torch.Tensor") -> "torch.Tensor":
    """Compute gradient of the Lovász extension w.r.t sorted errors.

    Args:
        gt_sorted: Ground truth indicators (0 or 1) sorted in descending order of error.

    Returns:
        Gradient vector delta.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # Covert jaccard to step differences
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(
    probas: "torch.Tensor",
    labels: "torch.Tensor",
    classes: Union[str, List[int]] = "present",
    ignore_index: Optional[int] = None,
) -> "torch.Tensor":
    """Multi-class Lovász-Softmax loss on flattened probability tensors.

    Args:
        probas: [P, C] Probabilities after softmax at each voxel/point.
        labels: [P] Ground truth class labels (0 .. C-1).
        classes: 'all' for all classes, 'present' for classes in labels, or list of class IDs.
        ignore_index: Class ID to exclude from loss calculation.

    Returns:
        Scalar Lovász-Softmax loss.
    """
    if probas.numel() == 0:
        return probas.new_tensor(0.0)

    C = probas.size(1)
    losses = []

    class_to_calc = list(range(C)) if classes == "all" else (classes if isinstance(classes, list) else None)

    for c in range(C):
        if ignore_index is not None and c == ignore_index:
            continue

        target_c = (labels == c).float()
        if class_to_calc is None:  # 'present'
            if target_c.sum() == 0:
                continue
        elif c not in class_to_calc:
            continue

        # Error for class c: 1 - prob if target is c, prob if target is not c
        prob_c = probas[:, c]
        errors = (target_c - prob_c).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        target_c_sorted = target_c[perm]

        grad = lovasz_grad(target_c_sorted)
        loss_c = torch.dot(errors_sorted, grad)
        losses.append(loss_c)

    if len(losses) == 0:
        return probas.new_tensor(0.0)

    return torch.stack(losses).mean()


def lovasz_softmax(
    probas: "torch.Tensor",
    labels: "torch.Tensor",
    classes: Union[str, List[int]] = "present",
    per_image: bool = False,
    ignore_index: Optional[int] = None,
) -> "torch.Tensor":
    """Lovász-Softmax loss entrypoint supporting 2D, 3D, and batched tensors.

    Args:
        probas: [N, C] or [B, C, N] or [B, N, C] Class probabilities.
        labels: [N] or [B, N] Ground truth class labels.
        classes: 'present' or 'all' or list of class IDs.
        per_image: Compute loss per image in batch if True.
        ignore_index: Optional class ID to ignore.

    Returns:
        Scalar tensor.
    """
    if probas.dim() == 3:
        # Check whether channel dimension is 1 or 2
        if probas.size(1) != labels.size(1):  # probas: [B, C, N], labels: [B, N]
            probas = probas.permute(0, 2, 1).contiguous()  # [B, N, C]
        B, N, C = probas.size()
        probas = probas.view(B * N, C)
        labels = labels.view(B * N)

    return lovasz_softmax_flat(probas, labels, classes=classes, ignore_index=ignore_index)


# =============================================================================
# Combined Segmentation Loss (Weighted CE + Lovász-Softmax)
# =============================================================================

class LovaszSoftmaxLoss(nn.Module if HAS_TORCH else object):
    """Stand-alone Lovász-Softmax loss module."""

    def __init__(
        self,
        classes: Union[str, List[int]] = "present",
        per_image: bool = False,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        """Compute Lovász loss on logits.

        Args:
            logits: (N, C) or (B, C, N) raw network predictions.
            targets: (N,) or (B, N) ground truth labels.
        """
        if logits.dim() == 3 and logits.size(1) != targets.size(1):
            # [B, C, N] -> softmax along channel dim 1
            probas = F.softmax(logits, dim=1)
        else:
            probas = F.softmax(logits, dim=-1)

        return lovasz_softmax(
            probas,
            targets,
            classes=self.classes,
            per_image=self.per_image,
            ignore_index=self.ignore_index,
        )


class CombinedLoss(nn.Module if HAS_TORCH else object):
    """Combined Weighted Cross-Entropy and Lovász-Softmax Loss.

    loss = ce_loss + lovasz_weight * lovasz_loss
    Configured from config/model_config.yaml training.loss block.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        num_classes: int = 6,
        class_weights: Optional[Union[List[float], "torch.Tensor"]] = None,
        lovasz_weight: float = 0.5,
        ignore_index: Optional[int] = None,
    ) -> None:
        """Initialize loss components.

        Args:
            config_path: Optional path to config/model_config.yaml.
            num_classes: Total semantic classes (default 6).
            class_weights: Per-class loss weights (higher for rare classes).
            lovasz_weight: Blend multiplier for Lovász-Softmax.
            ignore_index: Optional label index to ignore.
        """
        super().__init__()

        self.num_classes = num_classes
        self.lovasz_weight = lovasz_weight
        self.ignore_index = ignore_index

        default_weights = [1.0, 2.0, 3.0, 4.0, 5.0, 0.5]
        if class_weights is not None:
            self.class_weights_list = list(class_weights)
        else:
            self.class_weights_list = default_weights

        if config_path is not None:
            self.load_config(config_path)

        if HAS_TORCH:
            weights_tensor = torch.tensor(self.class_weights_list, dtype=torch.float32)
            self.register_buffer("class_weights", weights_tensor)
            self.ce_criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=self.ignore_index if self.ignore_index is not None else -100,
            )
            self.lovasz_criterion = LovaszSoftmaxLoss(
                classes="present",
                ignore_index=self.ignore_index,
            )

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load loss parameters from YAML configuration.

        Args:
            config_path: Path to model_config.yaml.
        """
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            logger.warning("Loss config path %s not found. Retaining defaults.", cfg_path)
            return

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        loss_cfg = cfg.get("training", {}).get("loss", {})
        if loss_cfg:
            if "class_weights" in loss_cfg:
                self.class_weights_list = list(loss_cfg["class_weights"])
            if "lovasz_weight" in loss_cfg:
                self.lovasz_weight = float(loss_cfg["lovasz_weight"])

    def forward(
        self, logits: "torch.Tensor", targets: "torch.Tensor"
    ) -> Dict[str, "torch.Tensor"]:
        """Compute combined loss.

        Args:
            logits: (N, C) or (B, C, N) or (B, N, C) network output logits.
            targets: (N,) or (B, N) target integer labels.

        Returns:
            Dictionary with 'loss', 'ce_loss', and 'lovasz_loss'.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to execute forward pass of CombinedLoss.")

        # Ensure shapes align for CrossEntropyLoss
        # If [B, C, N], CrossEntropy expects (B, C, N) with targets (B, N)
        # If [N, C], CrossEntropy expects (N, C) with targets (N,)
        ce_loss = self.ce_criterion(logits, targets)

        # Lovász loss
        if self.lovasz_weight > 0.0:
            lovasz_loss = self.lovasz_criterion(logits, targets)
        else:
            lovasz_loss = logits.new_tensor(0.0)

        total_loss = ce_loss + self.lovasz_weight * lovasz_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "lovasz_loss": lovasz_loss,
        }
