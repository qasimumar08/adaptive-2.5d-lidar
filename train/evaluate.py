"""Evaluation script for Lidar Semantic Segmentation models.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
Calculates:
  - Confusion Matrix (6x6)
  - Per-class IoU, Precision, Recall, F1-Score
  - Mean IoU (mIoU) and Overall Accuracy
  - Saves report to JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure workspace root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

import torch
import torch.nn as nn

from src.model.sparse_unet import SparseUNet
from src.model.pointnet2 import PointNet2SemSeg
try:
    from train.dataset_semantickitti import SemanticKITTIDataset, create_semantickitti_dataloader
except ImportError:
    from dataset_semantickitti import SemanticKITTIDataset, create_semantickitti_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASS_NAMES: Dict[int, str] = {
    0: "drivable_surface",
    1: "non_drivable_terrain",
    2: "static_obstacle",
    3: "dynamic_vehicle",
    4: "dynamic_pedestrian",
    5: "unknown_noise",
}


def compute_metrics_from_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Compute per-class IoU, Precision, Recall, F1 and mIoU from confusion matrix.

    Args:
        cm: Confusion matrix of shape (num_classes, num_classes) where rows are ground truth,
            and columns are predictions.
        class_names: Optional mapping from class index to readable class name.

    Returns:
        Dictionary containing overall and per-class metrics.
    """
    num_classes = cm.shape[0]
    names = class_names or CLASS_NAMES

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    ious: List[float] = []

    total_tp = 0
    total_samples = cm.sum()

    for c in range(num_classes):
        tp = float(cm[c, c])
        total_tp += tp
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        ious.append(iou)
        c_name = names.get(c, f"class_{c}")
        per_class_metrics[c_name] = {
            "class_id": c,
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": int(cm[c, :].sum()),
        }

    miou = float(np.mean(ious))
    accuracy = float(total_tp / total_samples) if total_samples > 0 else 0.0

    return {
        "mIoU": miou,
        "overall_accuracy": accuracy,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """Format evaluation metrics into an ASCII table.

    Args:
        metrics: Dictionary returned by compute_metrics_from_confusion_matrix.

    Returns:
        Formatted multi-line string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Class Name':<24} | {'IoU (%)':<10} | {'Prec (%)':<10} | {'Rec (%)':<10} | {'F1 (%)':<10} | {'Support':<8}")
    lines.append("-" * 80)

    for name, m in metrics["per_class"].items():
        lines.append(
            f"{name:<24} | {m['iou'] * 100:>9.2f}% | {m['precision'] * 100:>9.2f}% | "
            f"{m['recall'] * 100:>9.2f}% | {m['f1_score'] * 100:>9.2f}% | {m['support']:>8}"
        )

    lines.append("=" * 80)
    lines.append(f"Mean IoU (mIoU):     {metrics['mIoU'] * 100:.2f}%")
    lines.append(f"Overall Accuracy:   {metrics['overall_accuracy'] * 100:.2f}%")
    lines.append("=" * 80)
    return "\n".join(lines)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: Any,
    device: torch.device,
    model_type: str = "sparse_unet",
    num_classes: int = 6,
) -> Dict[str, Any]:
    """Evaluate trained model on dataloader.

    Args:
        model: PyTorch model.
        val_loader: PyTorch DataLoader.
        device: Torch device.
        model_type: 'sparse_unet' or 'pointnet2'.
        num_classes: Number of semantic classes.

    Returns:
        Evaluation metrics dictionary.
    """
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in val_loader:
        if model_type == "pointnet2":
            raw_items = batch.get("raw_batch", [])
            pts_list = [torch.from_numpy(it["points"]).float().to(device) for it in raw_items]
            lbl_list = [torch.from_numpy(it["labels"]).long().to(device) for it in raw_items if it.get("labels") is not None]

            if not pts_list:
                continue

            fixed_n = min(32768, min(len(p) for p in pts_list))
            pts_tensor = torch.stack([p[:fixed_n] for p in pts_list], dim=0)
            lbl_tensor = torch.stack([l[:fixed_n] for l in lbl_list], dim=0)

            logits = model(pts_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
            targets = lbl_tensor.cpu().numpy().flatten()
        else:
            sp_tensor = batch["sparse_tensor"]
            targets_t = batch["voxel_labels"]
            if targets_t is None or len(targets_t) == 0:
                continue

            logits = model(sp_tensor)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = targets_t.cpu().numpy()

        valid_mask = (targets >= 0) & (targets < num_classes)
        for t_val, p_val in zip(targets[valid_mask], preds[valid_mask]):
            confusion_matrix[t_val, p_val] += 1

    return compute_metrics_from_confusion_matrix(confusion_matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Semantic Segmentation Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to model config")
    parser.add_argument("--sensor-config", type=str, default="config/sensor_config.yaml", help="Path to sensor config")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root path")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")
    parser.add_argument("--output-json", type=str, default="eval_results.json", help="Path to save results JSON")

    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        logger.error("Checkpoint %s does not exist.", checkpoint_path)
        sys.exit(1)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model_name = checkpoint.get("model_name", "sparse_unet")

    logger.info("Loading %s from %s", model_name, checkpoint_path)
    if model_name == "pointnet2":
        model = PointNet2SemSeg.from_config(args.config).to(device)
    else:
        model = SparseUNet.from_config(args.config).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    dataset, dataloader = create_semantickitti_dataloader(
        model_config_path=args.config,
        sensor_config_path=args.sensor_config,
        split=args.split,
        data_root=args.data_root,
        batch_size=4,
        shuffle=False,
        device=device,
    )

    metrics = evaluate_model(model, dataloader, device, model_type=model_name)
    table = format_metrics_table(metrics)
    print(table)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", args.output_json)


if __name__ == "__main__":
    main()
