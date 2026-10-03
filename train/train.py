"""Training pipeline for Lidar Semantic Segmentation.

PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping.
Supports:
  - Sparse U-Net (spconv-triton) and PointNet++ models
  - AMD GPU / ROCm acceleration (torch.cuda.is_available())
  - Cosine LR scheduler with linear warmup
  - Gradient accumulation
  - TensorBoard logging (loss, mIoU, per-class IoU)
  - Best model checkpointing based on validation mIoU
  - Config loading from config/model_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore
        """Dummy SummaryWriter fallback."""
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def add_scalar(self, *args: Any, **kwargs: Any) -> None: pass
        def close(self) -> None: pass

from src.model.sparse_unet import SparseUNet
from src.model.pointnet2 import PointNet2SemSeg
from src.model.losses import CombinedLoss
from src.preprocessing.voxelizer import collate_sparse_voxels
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


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Select compute device with AMD ROCm GPU detection.

    Args:
        requested_device: Optional override ('cuda', 'cpu', 'cuda:0').

    Returns:
        torch.device.
    """
    if requested_device is not None:
        device = torch.device(requested_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        platform_info = f"ROCm HIP v{torch.version.hip}" if is_rocm else "CUDA"
        logger.info("Using GPU: %s (%s)", gpu_name, platform_info)
    else:
        logger.info("Using device: CPU")

    return device


def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 1e-4,
) -> LambdaLR:
    """Create a learning rate scheduler with linear warmup followed by cosine annealing.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Epochs for linear warmup.
        total_epochs: Total number of training epochs.
        min_lr_ratio: Minimum LR as a fraction of initial LR.

    Returns:
        LambdaLR scheduler instance.
    """
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Orchestrates model training, validation, checkpointing, and metrics tracking."""

    def __init__(
        self,
        model_name: str = "sparse_unet",
        config_path: Union[str, Path] = "config/model_config.yaml",
        sensor_config_path: Union[str, Path] = "config/sensor_config.yaml",
        data_root: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "./models",
        device: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.sensor_config_path = Path(sensor_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        train_cfg = self.cfg.get("training", {})
        self.total_epochs = epochs or int(train_cfg.get("epochs", 50))
        self.batch_size = batch_size or int(train_cfg.get("batch_size", 4))
        self.learning_rate = learning_rate or float(train_cfg.get("learning_rate", 0.001))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0001))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
        self.grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        self.val_interval = int(train_cfg.get("val_interval", 1))

        self.device = get_device(device)
        self.model_name = model_name.lower()

        # Initialize Model
        if self.model_name == "pointnet2":
            logger.info("Initializing PointNet++ fallback model.")
            self.model = PointNet2SemSeg.from_config(self.config_path).to(self.device)
        else:
            logger.info("Initializing spconv-triton Sparse U-Net model.")
            self.model = SparseUNet.from_config(self.config_path).to(self.device)

        # Initialize Loss
        self.criterion = CombinedLoss(
            config_path=self.config_path,
            num_classes=6,
        ).to(self.device)

        # Optimizer and Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.total_epochs,
        )

        # TensorBoard
        log_dir = self.output_dir / "logs"
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Dataset loaders
        self.train_dataset, self.train_loader = create_semantickitti_dataloader(
            model_config_path=self.config_path,
            sensor_config_path=self.sensor_config_path,
            split="train",
            data_root=data_root,
            batch_size=self.batch_size,
            shuffle=True,
            device=self.device,
        )
        self.val_dataset, self.val_loader = create_semantickitti_dataloader(
            model_config_path=self.config_path,
            sensor_config_path=self.sensor_config_path,
            split="val",
            data_root=data_root,
            batch_size=self.batch_size,
            shuffle=False,
            device=self.device,
        )

        self.best_miou = 0.0

    def train_epoch(self, epoch: int) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        steps = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            if self.model_name == "pointnet2":
                # For PointNet++, format points as (B, C, N)
                raw_items = batch.get("raw_batch", [])
                pts_list = [torch.from_numpy(it["points"]).float().to(self.device) for it in raw_items]
                lbl_list = [torch.from_numpy(it["labels"]).long().to(self.device) for it in raw_items if it.get("labels") is not None]

                # Subsample/pad points to fixed count if necessary
                fixed_n = 32768
                b_pts = []
                b_lbls = []
                for p, l in zip(pts_list, lbl_list):
                    if len(p) >= fixed_n:
                        idx = torch.randperm(len(p))[:fixed_n]
                        b_pts.append(p[idx])
                        b_lbls.append(l[idx])
                    else:
                        repeat_factor = (fixed_n // len(p)) + 1
                        p_rep = p.repeat(repeat_factor, 1)[:fixed_n]
                        l_rep = l.repeat(repeat_factor)[:fixed_n]
                        b_pts.append(p_rep)
                        b_lbls.append(l_rep)

                pts_tensor = torch.stack(b_pts, dim=0)  # (B, N, C)
                lbl_tensor = torch.stack(b_lbls, dim=0)  # (B, N)

                logits = self.model(pts_tensor)  # (B, num_classes, N)
                loss_dict = self.criterion(logits, lbl_tensor)
            else:
                sp_tensor = batch["sparse_tensor"]
                targets = batch["voxel_labels"]
                if targets is None or len(targets) == 0:
                    continue

                logits = self.model(sp_tensor)
                loss_dict = self.criterion(logits, targets)

            loss = loss_dict["loss"] / self.grad_accum_steps
            loss.backward()

            total_loss += loss_dict["loss"].item()
            steps += 1

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        avg_loss = total_loss / max(1, steps)
        current_lr = self.optimizer.param_groups[0]["lr"]

        self.writer.add_scalar("Train/Loss", avg_loss, epoch)
        self.writer.add_scalar("Train/LR", current_lr, epoch)

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, Dict[int, float]]:
        """Evaluate model on validation set, calculating per-class IoU and mIoU."""
        self.model.eval()
        total_loss = 0.0
        steps = 0

        num_classes = 6
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        for batch in self.val_loader:
            if self.model_name == "pointnet2":
                raw_items = batch.get("raw_batch", [])
                pts_list = [torch.from_numpy(it["points"]).float().to(self.device) for it in raw_items]
                lbl_list = [torch.from_numpy(it["labels"]).long().to(self.device) for it in raw_items if it.get("labels") is not None]

                fixed_n = min(32768, min(len(p) for p in pts_list)) if pts_list else 1000
                b_pts = [p[:fixed_n] for p in pts_list]
                b_lbls = [l[:fixed_n] for l in lbl_list]

                pts_tensor = torch.stack(b_pts, dim=0)
                lbl_tensor = torch.stack(b_lbls, dim=0)

                logits = self.model(pts_tensor)  # (B, num_classes, N)
                loss_dict = self.criterion(logits, lbl_tensor)
                preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
                targets = lbl_tensor.cpu().numpy().flatten()
            else:
                sp_tensor = batch["sparse_tensor"]
                targets_t = batch["voxel_labels"]
                if targets_t is None or len(targets_t) == 0:
                    continue

                logits = self.model(sp_tensor)
                loss_dict = self.criterion(logits, targets_t)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                targets = targets_t.cpu().numpy()

            total_loss += loss_dict["loss"].item()
            steps += 1

            # Accumulate confusion matrix
            mask = (targets >= 0) & (targets < num_classes)
            for t_val, p_val in zip(targets[mask], preds[mask]):
                confusion_matrix[t_val, p_val] += 1

        val_loss = total_loss / max(1, steps)

        # Compute IoU per class: TP / (TP + FP + FN)
        class_ious: Dict[int, float] = {}
        for c in range(num_classes):
            tp = confusion_matrix[c, c]
            fp = confusion_matrix[:, c].sum() - tp
            fn = confusion_matrix[c, :].sum() - tp
            denom = tp + fp + fn
            class_ious[c] = float(tp / denom) if denom > 0 else 0.0

        miou = float(np.mean(list(class_ious.values())))

        self.writer.add_scalar("Val/Loss", val_loss, epoch)
        self.writer.add_scalar("Val/mIoU", miou, epoch)
        for c, iou in class_ious.items():
            c_name = CLASS_NAMES.get(c, f"class_{c}")
            self.writer.add_scalar(f"Val_IoU/{c_name}", iou, epoch)

        return val_loss, miou, class_ious

    def save_checkpoint(self, filename: str, epoch: int, miou: float) -> None:
        """Save training state checkpoint."""
        save_path = self.output_dir / filename
        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_miou": miou,
            "config": self.cfg,
        }
        torch.save(checkpoint, str(save_path))
        logger.info("Saved checkpoint: %s (epoch %d, mIoU: %.2f%%)", save_path, epoch, miou * 100)

    def run(self) -> None:
        """Execute full training loop."""
        logger.info("Starting training %s for %d epochs.", self.model_name, self.total_epochs)

        if len(self.train_dataset) == 0:
            logger.warning(
                "Training dataset has 0 scans at data root: %s. "
                "Verify SemanticKITTI directory or run with synthetic samples.",
                self.train_dataset.data_root,
            )
            return

        for epoch in range(1, self.total_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            elapsed = time.time() - t0

            logger.info("Epoch [%d/%d] - Train Loss: %.4f (%.1fs)", epoch, self.total_epochs, train_loss, elapsed)

            if epoch % self.val_interval == 0:
                val_loss, val_miou, class_ious = self.validate(epoch)
                logger.info(
                    "Epoch [%d/%d] - Val Loss: %.4f | Val mIoU: %.2f%%",
                    epoch, self.total_epochs, val_loss, val_miou * 100
                )

                if val_miou > self.best_miou:
                    self.best_miou = val_miou
                    self.save_checkpoint("best_model.pt", epoch, self.best_miou)

                self.save_checkpoint("latest_model.pt", epoch, val_miou)

        logger.info("Training complete. Best validation mIoU: %.2f%%", self.best_miou * 100)
        self.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model (PS 26053)")
    parser.add_argument("--model", type=str, default="sparse_unet", choices=["sparse_unet", "pointnet2"], help="Model architecture")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to model config")
    parser.add_argument("--sensor-config", type=str, default="config/sensor_config.yaml", help="Path to sensor config")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root path")
    parser.add_argument("--output-dir", type=str, default="./models", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")

    args = parser.parse_args()

    trainer = Trainer(
        model_name=args.model,
        config_path=args.config,
        sensor_config_path=args.sensor_config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    trainer.run()


if __name__ == "__main__":
    main()
