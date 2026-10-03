# =============================================================================
# memory_benchmark.py — Memory Comparison: Foveated vs Uniform vs 3D Voxel
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BUILD_DIR = PROJECT_ROOT / "build"
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

try:
    import grid_py
    HAS_GRID_PY = True
except ImportError:
    grid_py = None
    HAS_GRID_PY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memory_benchmark")


# =============================================================================
# Memory Analysis & Theoretical Calculations
# =============================================================================

def compute_memory_profiles(
    grid_config_path: Union[str, Path] = "config/grid_params.yaml",
    cell_size_bytes: int = 24,
    voxel_size_bytes: int = 4,
    height_span_m: float = 6.0,
) -> Dict[str, Any]:
    """Compute detailed memory footprint comparing our foveated grid against baselines.

    Baselines:
      1. Foveated 2.5D Grid (Adaptive Variable Resolution)
      2. Uniform 5cm 2.5D Elevation Grid
      3. Uniform 3D Voxel Grid (5cm voxels)
    """
    grid_cfg_path = Path(grid_config_path)
    if not grid_cfg_path.is_absolute():
        grid_cfg_path = PROJECT_ROOT / grid_cfg_path

    with open(grid_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["grid"]

    max_range = float(cfg.get("max_range", 100.0))
    zones_cfg = cfg["zones"]

    # If grid_py is available, verify actual C++ sizeof(Cell)
    actual_cell_bytes = cell_size_bytes
    if HAS_GRID_PY:
        try:
            zm_test = grid_py.ZoneManager()
            zm_test.addZone("test", 0.0, 10.0, 0.05)
            zm_test.finalize()
            grid_test = grid_py.FoveatedGrid(zm_test)
            actual_cell_bytes = int(grid_test.memoryUsage() / grid_test.totalCells())
        except Exception as e:
            logger.warning(f"Could not verify C++ cell size via grid_py: {e}")

    # 1. Foveated Grid Zone Breakdown
    foveated_zones: List[Dict[str, Any]] = []
    total_foveated_cells = 0

    for z in zones_cfg:
        r_min = float(z["radius_min"])
        r_max = float(z["radius_max"])
        c_size = float(z["cell_size"])
        grid_dim = int(math.ceil((2.0 * r_max) / c_size))
        zone_cells = grid_dim * grid_dim
        zone_bytes = zone_cells * actual_cell_bytes

        foveated_zones.append({
            "name": z["name"],
            "radius_range_m": [r_min, r_max],
            "cell_size_m": c_size,
            "grid_dimensions": [grid_dim, grid_dim],
            "num_cells": zone_cells,
            "memory_bytes": zone_bytes,
            "memory_mb": zone_bytes / (1024.0 ** 2),
            "purpose": z.get("purpose", ""),
        })
        total_foveated_cells += zone_cells

    total_foveated_bytes = total_foveated_cells * actual_cell_bytes
    total_foveated_mb = total_foveated_bytes / (1024.0 ** 2)

    # 2. Uniform 5cm 2.5D Elevation Grid
    uniform_cell_size = 0.05  # Highest resolution across whole 100m range
    uniform_dim = int(math.ceil((2.0 * max_range) / uniform_cell_size))  # 4000
    total_uniform_cells = uniform_dim * uniform_dim  # 16,000,000
    total_uniform_bytes = total_uniform_cells * actual_cell_bytes
    total_uniform_mb = total_uniform_bytes / (1024.0 ** 2)

    # 3. Uniform 3D Voxel Grid (5cm voxels)
    voxel_z_dim = int(math.ceil(height_span_m / uniform_cell_size))  # 6.0 / 0.05 = 120
    total_3d_voxels = total_uniform_cells * voxel_z_dim  # 1.92 Billion voxels
    total_3d_bytes_compact = total_3d_voxels * voxel_size_bytes  # Compact 4-byte occupancy/label
    total_3d_mb_compact = total_3d_bytes_compact / (1024.0 ** 2)
    total_3d_bytes_full = total_3d_voxels * actual_cell_bytes  # Full cell metadata
    total_3d_mb_full = total_3d_bytes_full / (1024.0 ** 2)

    # Reductions
    reduction_vs_uniform_pct = ((total_uniform_cells - total_foveated_cells) / total_uniform_cells) * 100.0
    reduction_vs_3d_pct = ((total_3d_voxels - total_foveated_cells) / total_3d_voxels) * 100.0
    mem_factor_uniform = total_uniform_mb / total_foveated_mb
    mem_factor_3d = total_3d_mb_compact / total_foveated_mb

    report = {
        "cell_size_bytes": actual_cell_bytes,
        "height_span_m": height_span_m,
        "max_range_m": max_range,
        "foveated_grid": {
            "total_cells": total_foveated_cells,
            "memory_bytes": total_foveated_bytes,
            "memory_mb": float(total_foveated_mb),
            "zones": foveated_zones,
        },
        "uniform_2_5d_grid": {
            "cell_size_m": uniform_cell_size,
            "grid_dimensions": [uniform_dim, uniform_dim],
            "total_cells": total_uniform_cells,
            "memory_bytes": total_uniform_bytes,
            "memory_mb": float(total_uniform_mb),
        },
        "uniform_3d_voxel_grid": {
            "voxel_size_m": [uniform_cell_size, uniform_cell_size, uniform_cell_size],
            "grid_dimensions": [uniform_dim, uniform_dim, voxel_z_dim],
            "total_voxels": total_3d_voxels,
            "compact_memory_mb": float(total_3d_mb_compact),
            "full_memory_mb": float(total_3d_mb_full),
        },
        "savings": {
            "reduction_vs_uniform_2_5d_pct": float(reduction_vs_uniform_pct),
            "reduction_vs_3d_voxel_pct": float(reduction_vs_3d_pct),
            "memory_reduction_factor_vs_uniform": float(mem_factor_uniform),
            "memory_reduction_factor_vs_3d_compact": float(mem_factor_3d),
        },
    }

    return report


# =============================================================================
# Matplotlib Visualization
# =============================================================================

def generate_memory_charts(report: Dict[str, Any], output_path: Path) -> None:
    """Generate high-impact comparison visualizations of memory savings."""
    fov = report["foveated_grid"]
    uni = report["uniform_2_5d_grid"]
    vox = report["uniform_3d_voxel_grid"]
    sav = report["savings"]

    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # -------------------------------------------------------------------------
    # Panel 1: Memory Footprint (MB / GB) Comparison
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ["Foveated 2.5D\n(Our Approach)", "Uniform 5cm 2.5D\n(Baseline)", "3D Voxel Grid\n(Compact 4B)"]
    mbs = [fov["memory_mb"], uni["memory_mb"], vox["compact_memory_mb"]]
    colors = ["#2a9d8f", "#e76f51", "#264653"]

    bars1 = ax1.bar(categories, mbs, color=colors, edgecolor="black", width=0.55)
    ax1.set_ylabel("RAM Memory Footprint (Megabytes)", fontsize=11, fontweight="bold")
    ax1.set_title("Memory Consumption Comparison", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate values
    for b in bars1:
        val = b.get_height()
        if val >= 1024.0:
            text = f"{val / 1024.0:.2f} GB\n({val:.0f} MB)"
        else:
            text = f"{val:.1f} MB"
        ax1.annotate(text, xy=(b.get_x() + b.get_width() / 2, val), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Savings badge
    ax1.text(0.18, 0.75, f"{sav['reduction_vs_uniform_2_5d_pct']:.1f}% RAM Reduction\n({sav['memory_reduction_factor_vs_uniform']:.1f}x Smaller)",
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="#d8f3dc", edgecolor="#2d6a4f"),
             fontsize=10, fontweight="bold", color="#1b4332")

    # -------------------------------------------------------------------------
    # Panel 2: Total Cell Count (Logarithmic Scale)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    counts = [fov["total_cells"], uni["total_cells"], vox["total_voxels"]]
    bars2 = ax2.bar(categories, counts, color=["#588157", "#f4a261", "#1d3557"], edgecolor="black", width=0.55)
    ax2.set_yscale("log")
    ax2.set_ylabel("Total Number of Cells / Voxels (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("Cell Count Complexity (Log10)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for b, c in zip(bars2, counts):
        if c >= 1e9:
            txt = f"{c / 1e9:.2f} B"
        elif c >= 1e6:
            txt = f"{c / 1e6:.2f} M"
        else:
            txt = f"{c / 1e3:.1f} k"
        ax2.annotate(txt, xy=(b.get_x() + b.get_width() / 2, c), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: Foveated Zone Memory Distribution (Donut Chart)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    zone_names = [f"{z['name'].capitalize()}\n({z['cell_size_m'] * 100:.0f}cm, {z['radius_range_m'][0]:.0f}-{z['radius_range_m'][1]:.0f}m)" for z in fov["zones"]]
    zone_cells = [z["num_cells"] for z in fov["zones"]]
    zone_colors = ["#3a86ff", "#8338ec", "#ff006e", "#fb5607"]

    wedges, texts, autotexts = ax3.pie(
        zone_cells,
        labels=zone_names,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=140,
        colors=zone_colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax3.set_title(f"Foveated Grid Cell Allocation by Zone\n(Total: {fov['total_cells']:,} cells = {fov['memory_mb']:.1f} MB)", fontsize=12, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 4: Spatial Resolution vs Radial Distance Curve
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    r_sampled = np.linspace(0.0, 100.0, 500)
    fov_res = []
    for r in r_sampled:
        if r < 10.0:
            fov_res.append(0.05)
        elif r < 30.0:
            fov_res.append(0.10)
        elif r < 60.0:
            fov_res.append(0.25)
        else:
            fov_res.append(0.50)

    ax4.step(r_sampled, np.array(fov_res) * 100.0, where="post", color="#e63946", linewidth=2.5, label="Foveated Grid Resolution (cm)")
    ax4.axhline(5.0, color="#457b9d", linestyle="--", linewidth=1.8, label="Uniform 5cm Baseline")

    # Shade zones
    ax4.axvspan(0.0, 10.0, alpha=0.10, color="blue", label="Immediate (0-10m)")
    ax4.axvspan(10.0, 30.0, alpha=0.10, color="purple", label="Near (10-30m)")
    ax4.axvspan(30.0, 60.0, alpha=0.10, color="magenta", label="Mid (30-60m)")
    ax4.axvspan(60.0, 100.0, alpha=0.10, color="orange", label="Far (60-100m)")

    ax4.set_xlabel("Radial Distance from Ego-Vehicle (meters)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Cell Resolution Size (cm)", fontsize=11, fontweight="bold")
    ax4.set_title("Adaptive Variable Resolution Profile", fontsize=12, fontweight="bold")
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 55)
    ax4.legend(loc="upper left", fontsize=8.5)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.32, wspace=0.25)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved memory comparison chart to: {output_path}")


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Memory Benchmark for Foveated 2.5D Grid Engine")
    parser.add_argument("--config", type=str, default="config/grid_params.yaml", help="Path to grid params config")
    parser.add_argument("--output-json", type=str, default="benchmarks/results/memory_benchmark.json")
    parser.add_argument("--output-chart", type=str, default="benchmarks/results/memory_comparison.png")
    args = parser.parse_args()

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_chart = Path(args.output_chart)
    out_chart.parent.mkdir(parents=True, exist_ok=True)

    report = compute_memory_profiles(args.config)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved memory report to: {out_json}")

    generate_memory_charts(report, out_chart)

    fov = report["foveated_grid"]
    uni = report["uniform_2_5d_grid"]
    vox = report["uniform_3d_voxel_grid"]
    sav = report["savings"]

    print("\n" + "=" * 75)
    print("           COMPREHENSIVE MEMORY BENCHMARK REPORT (PS 26053)")
    print("=" * 75)
    print(f"Cell Struct Size:     {report['cell_size_bytes']} bytes")
    print(f"Lidar Max Range:      {report['max_range_m']} meters")
    print(f"3D Height Span:       {report['height_span_m']} meters")
    print("-" * 75)
    print(f"{'Representation':<26} | {'Total Cells/Voxels':<18} | {'Memory (RAM)':<18}")
    print("-" * 75)
    print(f"{'1. Foveated 2.5D (Ours)':<26} | {fov['total_cells']:<18,d} | {fov['memory_mb']:<10.2f} MB")
    print(f"{'2. Uniform 5cm 2.5D':<26} | {uni['total_cells']:<18,d} | {uni['memory_mb']:<10.2f} MB")
    print(f"{'3. 3D Voxel Grid (4B compact)':<26} | {vox['total_voxels']:<18,d} | {vox['compact_memory_mb']:<10.2f} MB ({vox['compact_memory_mb']/1024:.2f} GB)")
    print(f"{'4. 3D Voxel Grid (24B full)':<26} | {vox['total_voxels']:<18,d} | {vox['full_memory_mb']:<10.2f} MB ({vox['full_memory_mb']/1024:.2f} GB)")
    print("-" * 75)
    print(f"Memory Reduction vs Uniform 2.5D:  {sav['reduction_vs_uniform_2_5d_pct']:.2f}%  ({sav['memory_reduction_factor_vs_uniform']:.1f}x reduction)")
    print(f"Memory Reduction vs 3D Voxel Grid: {sav['reduction_vs_3d_voxel_pct']:.4f}% ({sav['memory_reduction_factor_vs_3d_compact']:.1f}x reduction)")
    print("-" * 75)
    print("Foveated Zone Breakdown:")
    for z in fov["zones"]:
        print(f"  • {z['name']:<10} ({z['radius_range_m'][0]:.0f}-{z['radius_range_m'][1]:.0f}m, res {z['cell_size_m']*100:.0f}cm): {z['num_cells']:,} cells ({z['memory_mb']:.2f} MB)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
