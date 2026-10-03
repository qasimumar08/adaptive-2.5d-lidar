# =============================================================================
# test_benchmarks_and_demo.py — Unit Tests for Benchmarks, ONNX, and Demo
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.model.export_onnx import (
    DenseEquivalentSparseUNet,
    export_model_to_onnx,
    create_onnxruntime_session,
    benchmark_pytorch_vs_onnx,
    HAS_ONNX,
)
from benchmarks.latency_benchmark import (
    GPUTimer,
    generate_synthetic_scan,
    run_latency_benchmark,
)
from benchmarks.memory_benchmark import (
    compute_memory_profiles,
    generate_memory_charts,
)
from benchmarks.accuracy_vs_distance import (
    compute_confusion_matrix,
    compute_metrics_from_cm,
    evaluate_accuracy_vs_distance,
)
from demo.run_demo import (
    DynamicDrivingSimulator,
    run_end_to_end_demo,
)


# -----------------------------------------------------------------------------
# ONNX Export Tests
# -----------------------------------------------------------------------------

def test_dense_equivalent_sparse_unet_forward():
    model = DenseEquivalentSparseUNet(
        in_channels=4,
        num_classes=6,
        encoder_channels=[8, 16, 32, 64, 128],
        decoder_channels=[128, 64, 32, 16, 8],
    )
    model.eval()
    # Spatial dimensions must be multiples of 16 for 4 downsamplings
    x = torch.randn(1, 4, 16, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 16, 16, 16)


@pytest.mark.skipif(not HAS_ONNX, reason="onnx / onnxruntime not installed")
def test_onnx_export_and_session(tmp_path):
    model = DenseEquivalentSparseUNet(
        in_channels=4,
        num_classes=6,
        encoder_channels=[8, 16, 32, 64, 128],
        decoder_channels=[128, 64, 32, 16, 8],
    )
    dummy_input = torch.randn(1, 4, 16, 16, 16)
    onnx_file = tmp_path / "test_model.onnx"
    mxr_file = tmp_path / "test_model.mxr"

    out_p = export_model_to_onnx(
        model=model,
        dummy_input=dummy_input,
        output_path=onnx_file,
        opset_version=17,
    )
    assert out_p.exists()
    assert out_p.stat().st_size > 0

    session, provider = create_onnxruntime_session(
        onnx_path=onnx_file,
        mxr_cache_path=mxr_file,
        device="cpu",
    )
    assert session is not None
    assert mxr_file.exists()

    metrics = benchmark_pytorch_vs_onnx(
        torch_model=model,
        ort_session=session,
        dummy_input=dummy_input,
        num_runs=5,
        warmup_runs=2,
        device="cpu",
    )
    assert "pytorch" in metrics
    assert "onnxruntime" in metrics
    assert metrics["speedup_ratio"] > 0.0


# -----------------------------------------------------------------------------
# Latency Benchmark Tests
# -----------------------------------------------------------------------------

def test_gpu_timer_cpu_fallback():
    timer = GPUTimer(device="cpu")
    timer.start()
    # Busy wait small duration
    _ = [x ** 2 for x in range(10000)]
    elapsed = timer.stop()
    assert elapsed > 0.0


def test_synthetic_scan_generation():
    scan = generate_synthetic_scan(num_points=5000)
    assert scan.shape == (5000, 4)
    assert np.all(np.isfinite(scan))


def test_latency_benchmark_fast(tmp_path):
    res = run_latency_benchmark(
        num_frames=3,
        warmup_frames=1,
        points_per_frame=2000,
        device="cpu",
        output_dir=tmp_path,
    )
    assert res["num_frames"] == 3
    assert res["fps"] > 0.0
    assert (tmp_path / "latency_benchmark.json").exists()
    assert (tmp_path / "latency_breakdown.png").exists()
    assert (tmp_path / "latency_distribution.png").exists()


# -----------------------------------------------------------------------------
# Memory Benchmark Tests
# -----------------------------------------------------------------------------

def test_memory_profile_computations(tmp_path):
    report = compute_memory_profiles()
    assert report["cell_size_bytes"] == 24
    assert report["foveated_grid"]["total_cells"] == 910400
    assert report["uniform_2_5d_grid"]["total_cells"] == 16000000
    assert report["savings"]["reduction_vs_uniform_2_5d_pct"] > 94.0

    chart_file = tmp_path / "mem_test.png"
    generate_memory_charts(report, chart_file)
    assert chart_file.exists()
    assert chart_file.stat().st_size > 0


# -----------------------------------------------------------------------------
# Accuracy vs Distance Tests
# -----------------------------------------------------------------------------

def test_confusion_matrix_and_metrics():
    y_true = np.array([0, 0, 1, 2, 3, 4], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 2, 3, 4], dtype=np.int64)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=6)
    assert cm.shape == (6, 6)
    assert cm[0, 0] == 1
    assert cm[0, 1] == 1

    metrics = compute_metrics_from_cm(cm)
    assert metrics["mIoU"] > 0.0
    assert metrics["mean_f1"] > 0.0


def test_accuracy_vs_distance_fast(tmp_path):
    report = evaluate_accuracy_vs_distance(
        num_eval_frames=2,
        device="cpu",
        output_dir=tmp_path,
    )
    assert len(report["zones"]) == 4
    assert (tmp_path / "accuracy_vs_distance.json").exists()
    assert (tmp_path / "accuracy_vs_distance.png").exists()


# -----------------------------------------------------------------------------
# Demo Simulation Tests
# -----------------------------------------------------------------------------

def test_dynamic_driving_simulator():
    sim = DynamicDrivingSimulator(num_points_per_frame=1000)
    pts, meta = sim.get_next_frame()
    assert pts.shape[1] == 4
    assert meta["frame_idx"] == 1
    assert meta["ego_dist_m"] >= 0.0


def test_run_end_to_end_demo_fast(tmp_path):
    video_out = tmp_path / "demo_test.mp4"
    summary = run_end_to_end_demo(
        kitti_seq=None,
        num_frames=3,
        output_video=str(video_out),
        headless=True,
        fps_target=10,
        device="cpu",
    )
    assert summary["frames_processed"] == 3
    assert video_out.exists()
    assert video_out.stat().st_size > 0
    assert video_out.with_suffix(".json").exists()
