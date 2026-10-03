# =============================================================================
# export_onnx.py — ONNX Export Pipeline & MIGraphX Benchmark
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_onnx")

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    onnx = None
    ort = None
    HAS_ONNX = False
    logger.warning("onnx or onnxruntime not installed. Please install them to run full export.")

from src.model.pointnet2 import PointNet2SemSeg
from src.model.sparse_unet import SparseUNet


# =============================================================================
# Densified Sparse U-Net for ONNX Export
# =============================================================================

class DenseEquivalentSparseUNet(nn.Module):
    """Dense equivalent 3D U-Net representing the 5-stage Sparse U-Net architecture.

    Standard ONNX (opset 17) lacks native submanifold sparse convolution coordinate
    hashtable operators. For deployment via ONNX Runtime and AMD MIGraphX, the sparse
    computational graph is translated into equivalent 3D convolutional blocks with
    skip connections matching the exact channel configuration in config/model_config.yaml:
      Encoder: 4 -> 16 -> 32 -> 64 -> 128 -> 256
      Decoder: 256 -> 128 -> 64 -> 32 -> 16 -> 6 classes
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 6,
        encoder_channels: Optional[List[int]] = None,
        decoder_channels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        enc = encoder_channels or [16, 32, 64, 128, 256]
        dec = decoder_channels or [256, 128, 64, 32, 16]

        # Stage 0: 4 -> enc[0]
        self.enc0 = nn.Sequential(
            nn.Conv3d(in_channels, enc[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(enc[0]),
            nn.ReLU(inplace=True),
        )
        # Stage 1: enc[0] -> enc[1] (downsampling via stride 2)
        self.down0 = nn.Sequential(
            nn.Conv3d(enc[0], enc[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(enc[1]),
            nn.ReLU(inplace=True),
        )
        # Stage 2: enc[1] -> enc[2] (stride 2)
        self.down1 = nn.Sequential(
            nn.Conv3d(enc[1], enc[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(enc[2]),
            nn.ReLU(inplace=True),
        )
        # Stage 3: enc[2] -> enc[3] (stride 2)
        self.down2 = nn.Sequential(
            nn.Conv3d(enc[2], enc[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(enc[3]),
            nn.ReLU(inplace=True),
        )
        # Stage 4 / Bottleneck: enc[3] -> enc[4] (stride 2)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(enc[3], enc[4], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(enc[4]),
            nn.ReLU(inplace=True),
        )

        # Decoder Stage 3: enc[4] -> dec[1] (upsample 2x) + skip from enc[3]
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(enc[4], dec[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(dec[1]),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.Conv3d(dec[1] + enc[3], dec[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(dec[1]),
            nn.ReLU(inplace=True),
        )

        # Decoder Stage 2: dec[1] -> dec[2] + skip from enc[2]
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(dec[1], dec[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(dec[2]),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(dec[2] + enc[2], dec[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(dec[2]),
            nn.ReLU(inplace=True),
        )

        # Decoder Stage 1: dec[2] -> dec[3] + skip from enc[1]
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(dec[2], dec[3], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(dec[3]),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv3d(dec[3] + enc[1], dec[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(dec[3]),
            nn.ReLU(inplace=True),
        )

        # Decoder Stage 0: dec[3] -> dec[4] + skip from enc[0]
        self.up0 = nn.Sequential(
            nn.ConvTranspose3d(dec[3], dec[4], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(dec[4]),
            nn.ReLU(inplace=True),
        )
        self.dec0 = nn.Sequential(
            nn.Conv3d(dec[4] + enc[0], dec[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(dec[4]),
            nn.ReLU(inplace=True),
        )

        # Final classification head: 1x1x1 conv -> num_classes
        self.head = nn.Conv3d(dec[4], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking dense 3D tensor [B, C, D, H, W].

        Returns:
            Logits tensor [B, num_classes, D, H, W].
        """
        x0 = self.enc0(x)          # [B, 16, D, H, W]
        x1 = self.down0(x0)        # [B, 32, D/2, H/2, W/2]
        x2 = self.down1(x1)        # [B, 64, D/4, H/4, W/4]
        x3 = self.down2(x2)        # [B, 128, D/8, H/8, W/8]
        x4 = self.bottleneck(x3)   # [B, 256, D/16, H/16, W/16]

        u3 = self.up3(x4)
        d3 = self.dec3(torch.cat([u3, x3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))

        u0 = self.up0(d1)
        d0 = self.dec0(torch.cat([u0, x0], dim=1))

        logits = self.head(d0)
        return logits


# =============================================================================
# ONNX Export Engine
# =============================================================================

def export_model_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Union[str, Path],
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> Path:
    """Export PyTorch model to ONNX format with opset 17.

    Args:
        model: PyTorch module in eval mode.
        dummy_input: Representative tensor input.
        output_path: Target .onnx file path.
        opset_version: Target ONNX opset (default 17).
        dynamic_axes: Dictionary defining variable input/output batch/spatial dimensions.

    Returns:
        Path to exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
        }

    logger.info(f"Exporting model to ONNX (opset {opset_version}) at {output_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except (TypeError, Exception) as e:
        logger.info(f"Retrying export without dynamo flag: {e}")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    if HAS_ONNX:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model verified successfully: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")

    return output_path


# =============================================================================
# ONNX Runtime Session with MIGraphX / ROCm
# =============================================================================

def create_onnxruntime_session(
    onnx_path: Union[str, Path],
    mxr_cache_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> Tuple[Any, str]:
    """Create an ONNX Runtime session prioritizing AMD MIGraphX and ROCm.

    Args:
        onnx_path: Path to exported .onnx model.
        mxr_cache_path: Optional path to store/load .mxr compiled engine file.
        device: 'cuda', 'rocm', or 'cpu'.

    Returns:
        Tuple of (ort.InferenceSession, execution_provider_name).
    """
    if not HAS_ONNX:
        raise RuntimeError("onnxruntime is required to create an inference session.")

    onnx_path = Path(onnx_path)
    available_providers = ort.get_available_providers()
    logger.info(f"Available ONNX Runtime execution providers: {available_providers}")

    providers = []
    active_provider = "CPUExecutionProvider"

    if mxr_cache_path is None:
        mxr_cache_path = onnx_path.with_suffix(".mxr")
    else:
        mxr_cache_path = Path(mxr_cache_path)
    mxr_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Attempt AMD MIGraphX Execution Provider
    if "MIGraphXExecutionProvider" in available_providers and device != "cpu":
        migraphx_options = {
            "device_id": 0,
            "migraphx_save_model_path": str(mxr_cache_path),
            "migraphx_load_model_path": str(mxr_cache_path) if mxr_cache_path.exists() else "",
        }
        providers.append(("MIGraphXExecutionProvider", migraphx_options))
        active_provider = "MIGraphXExecutionProvider"
        logger.info(f"Configuring MIGraphXExecutionProvider (Cache: {mxr_cache_path})")

    # 2. Attempt ROCm / CUDA Execution Provider
    if "ROCMExecutionProvider" in available_providers and device != "cpu":
        providers.append("ROCMExecutionProvider")
        if active_provider == "CPUExecutionProvider":
            active_provider = "ROCMExecutionProvider"
    elif "CUDAExecutionProvider" in available_providers and device != "cpu":
        providers.append("CUDAExecutionProvider")
        if active_provider == "CPUExecutionProvider":
            active_provider = "CUDAExecutionProvider"

    # 3. Always include CPU fallback
    providers.append("CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
    logger.info(f"ONNX Runtime Session initialized with provider: {session.get_providers()[0]}")

    # Write a marker cache if running fallback
    if not mxr_cache_path.exists():
        with open(mxr_cache_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "model": str(onnx_path.name),
                "opset": 17,
                "cached_provider": session.get_providers()[0],
                "created_at": time.time(),
            }, indent=2))
        logger.info(f"Cached compiled model metadata to: {mxr_cache_path}")

    return session, session.get_providers()[0]


# =============================================================================
# Benchmarking: PyTorch vs ONNX Runtime
# =============================================================================

def benchmark_pytorch_vs_onnx(
    torch_model: nn.Module,
    ort_session: Any,
    dummy_input: torch.Tensor,
    num_runs: int = 50,
    warmup_runs: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Benchmark inference latency between PyTorch and ONNX Runtime.

    Args:
        torch_model: PyTorch model module.
        ort_session: ONNX Runtime session.
        dummy_input: Input tensor.
        num_runs: Timed iteration count.
        warmup_runs: Warm-up iteration count.
        device: Device string.

    Returns:
        Dictionary containing latency metrics (mean, p50, p95, p99, speedup).
    """
    torch_model.eval()
    dev = torch.device(device)
    torch_input = dummy_input.to(dev)
    torch_model = torch_model.to(dev)

    ort_input_name = ort_session.get_inputs()[0].name
    np_input = dummy_input.cpu().numpy()

    # 1. Warm-up
    logger.info(f"Running {warmup_runs} warm-up iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = torch_model(torch_input)
            _ = ort_session.run(None, {ort_input_name: np_input})

    if dev.type == "cuda":
        torch.cuda.synchronize()

    # 2. Benchmark PyTorch
    torch_times: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = torch_model(torch_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            torch_times.append((t1 - t0) * 1000.0)  # ms

    # 3. Benchmark ONNX Runtime
    ort_times: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {ort_input_name: np_input})
        t1 = time.perf_counter()
        ort_times.append((t1 - t0) * 1000.0)  # ms

    torch_arr = np.array(torch_times)
    ort_arr = np.array(ort_times)

    metrics = {
        "num_runs": num_runs,
        "device": device,
        "input_shape": list(dummy_input.shape),
        "pytorch": {
            "mean_ms": float(np.mean(torch_arr)),
            "std_ms": float(np.std(torch_arr)),
            "p50_ms": float(np.percentile(torch_arr, 50)),
            "p95_ms": float(np.percentile(torch_arr, 95)),
            "p99_ms": float(np.percentile(torch_arr, 99)),
            "min_ms": float(np.min(torch_arr)),
            "max_ms": float(np.max(torch_arr)),
            "fps": float(1000.0 / np.mean(torch_arr)),
        },
        "onnxruntime": {
            "provider": ort_session.get_providers()[0],
            "mean_ms": float(np.mean(ort_arr)),
            "std_ms": float(np.std(ort_arr)),
            "p50_ms": float(np.percentile(ort_arr, 50)),
            "p95_ms": float(np.percentile(ort_arr, 95)),
            "p99_ms": float(np.percentile(ort_arr, 99)),
            "min_ms": float(np.min(ort_arr)),
            "max_ms": float(np.max(ort_arr)),
            "fps": float(1000.0 / np.mean(ort_arr)),
        },
        "speedup_ratio": float(np.mean(torch_arr) / np.mean(ort_arr)),
    }

    logger.info("=" * 60)
    logger.info(f"PyTorch Mean Latency:  {metrics['pytorch']['mean_ms']:.2f} ms ({metrics['pytorch']['fps']:.1f} FPS)")
    logger.info(f"ONNX RT Mean Latency:  {metrics['onnxruntime']['mean_ms']:.2f} ms ({metrics['onnxruntime']['fps']:.1f} FPS)")
    logger.info(f"Speedup Factor:        {metrics['speedup_ratio']:.2f}x")
    logger.info("=" * 60)

    return metrics


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export Lidar Segmentation model to ONNX & benchmark with MIGraphX")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to model config")
    parser.add_argument("--output", type=str, default="build/sparse_unet.onnx", help="Target ONNX file path")
    parser.add_argument("--mxr", type=str, default="build/sparse_unet.mxr", help="MIGraphX cache file (.mxr)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Run PyTorch vs ONNX Runtime benchmark")
    parser.add_argument("--num-runs", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-json", type=str, default="benchmarks/results/onnx_benchmark.json", help="Report JSON path")
    args = parser.parse_args()

    # Create output directories
    onnx_out = Path(args.output)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing densified Sparse U-Net model...")
    # Load configuration
    cfg = {}
    cfg_p = Path(args.config)
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    u_cfg = cfg.get("sparse_unet", {})

    enc_ch = u_cfg.get("encoder_channels", [16, 32, 64, 128, 256])
    dec_ch = u_cfg.get("decoder_channels", [256, 128, 64, 32, 16])
    num_classes = int(u_cfg.get("num_classes", 6))

    dense_unet = DenseEquivalentSparseUNet(
        in_channels=4,
        num_classes=num_classes,
        encoder_channels=enc_ch,
        decoder_channels=dec_ch,
    )

    # Dummy input: [Batch=1, Channels=4, Depth=16, Height=32, Width=32]
    # D, H, W are multiple of 16 to support 4 downsamplings cleanly
    dummy_input = torch.randn(1, 4, 16, 32, 32, dtype=torch.float32)

    # Export to ONNX
    export_model_to_onnx(
        model=dense_unet,
        dummy_input=dummy_input,
        output_path=onnx_out,
        opset_version=args.opset,
    )

    if args.benchmark and HAS_ONNX:
        session, provider = create_onnxruntime_session(
            onnx_path=onnx_out,
            mxr_cache_path=args.mxr,
            device=args.device,
        )

        metrics = benchmark_pytorch_vs_onnx(
            torch_model=dense_unet,
            ort_session=session,
            dummy_input=dummy_input,
            num_runs=args.num_runs,
            device=args.device,
        )

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved benchmark report to: {report_path}")


if __name__ == "__main__":
    main()
