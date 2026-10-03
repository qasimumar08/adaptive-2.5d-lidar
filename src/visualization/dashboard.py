# =============================================================================
# dashboard.py — Real-Time Dear ImGui Visualization Dashboard (OpenGL / AMD)
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BUILD_DIR = PROJECT_ROOT / "build"
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

try:
    import grid_py
except ImportError:
    grid_py = None

from src.visualization.colormap import Colormap, default_colormap

# Check if DearPyGui (Dear ImGui with OpenGL backend) is available
try:
    import dearpygui.dearpygui as dpg
    HAS_DPG = True
except ImportError:
    HAS_DPG = False


class GridDashboard:
    """
    Real-Time Dear ImGui Dashboard with OpenGL rendering for AMD GPUs.
    Visualizes:
      1. Composite Top-Down 2.5D Map (OpenCV rendered into OpenGL texture)
      2. Concentric Zone Boundaries (10m, 30m, 60m, 100m)
      3. Real-time FPS & Latency Histogram
      4. Memory Footprint Comparison (Foveated vs Uniform 2.5D vs 3D Voxels)
      5. Per-Class Semantic Cell Distribution
    """

    def __init__(
        self,
        canvas_size: int = 600,
        max_range: float = 100.0,
        colormap: Optional[Colormap] = None,
        headless: bool = False,
    ):
        self.canvas_size = canvas_size
        self.max_range = max_range
        self.colormap = colormap or default_colormap
        self.headless = headless or ("DISPLAY" not in os.environ)

        # Performance history buffers
        self.latency_history = deque(maxlen=100)
        self.fps_history = deque(maxlen=100)
        self.last_frame_time = time.time()

        # Metrics cache
        self.current_fps = 0.0
        self.current_latency_ms = 0.0
        self.memory_stats: Dict[str, float] = {}
        self.class_counts: Dict[int, int] = {c: 0 for c in range(6)}

        # Texture buffer for DearPyGui (RGBA float format, shape: (canvas_size * canvas_size * 4))
        self.raw_texture_data = np.zeros((self.canvas_size, self.canvas_size, 4), dtype=np.float32)

        self._is_dpg_initialized = False
        if not self.headless and HAS_DPG:
            self._init_dpg_window()

    def _init_dpg_window(self) -> None:
        """Initialize DearPyGui / Dear ImGui OpenGL window and layouts."""
        dpg.create_context()

        # Register dynamic texture for top-down 2.5D map
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=self.canvas_size,
                height=self.canvas_size,
                default_value=self.raw_texture_data.ravel(),
                tag="texture_2d_map",
            )


        with dpg.window(label="PS 26053 — Foveated 2.5D Lidar Dashboard (AMD ROCm / OpenGL)", tag="Primary Window"):
            with dpg.group(horizontal=True):
                # Left Column: 2.5D Top-Down Viewport
                with dpg.child_window(width=self.canvas_size + 20, height=self.canvas_size + 80):
                    dpg.add_text("Top-Down 2.5D Elevation & Semantic Grid", color=[255, 215, 0])
                    dpg.add_image("texture_2d_map", width=self.canvas_size, height=self.canvas_size)
                    dpg.add_text("Concentric Rings: Immediate (10m) | Near (30m) | Mid (60m) | Far (100m)")

                # Right Column: Metrics, Memory, and Semantic Class Stats
                with dpg.child_window(width=420, height=self.canvas_size + 80):
                    dpg.add_text("System Telemetry & Performance", color=[0, 255, 200])
                    dpg.add_separator()

                    dpg.add_text("FPS: 0.0", tag="txt_fps")
                    dpg.add_text("Latency: 0.0 ms", tag="txt_latency")

                    dpg.add_spacer(height=8)
                    dpg.add_text("Latency Histogram (Last 100 Frames)", color=[200, 200, 255])
                    with dpg.plot(label="Latency (ms)", height=120, width=-1):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Frame", no_tick_labels=True)
                        with dpg.plot_axis(dpg.mvYAxis, label="ms"):
                            dpg.add_line_series([], [], label="Total Pipeline Latency", tag="series_latency")

                    dpg.add_spacer(height=8)
                    dpg.add_text("Memory Footprint Comparison", color=[255, 180, 50])
                    dpg.add_separator()
                    dpg.add_text("Foveated Grid:  Calculating...", tag="txt_mem_foveated")
                    dpg.add_text("Uniform 5cm:    Calculating...", tag="txt_mem_uniform")
                    dpg.add_text("3D Voxel Grid:  Calculating...", tag="txt_mem_3d")
                    dpg.add_text("Memory Savings: Calculating...", color=[0, 255, 100], tag="txt_mem_savings")

                    dpg.add_spacer(height=8)
                    dpg.add_text("Semantic Cell Distribution", color=[100, 200, 255])
                    dpg.add_separator()
                    for cid in range(6):
                        cname = self.colormap.get_class_name(cid)
                        r, g, b = self.colormap.get_color_rgb(cid)
                        dpg.add_text(f"{cname}: 0 cells", color=[r, g, b], tag=f"txt_class_{cid}")

        dpg.create_viewport(
            title="PS 26053 — Foveated 2.5D Lidar Mapping Dashboard",
            width=self.canvas_size + 480,
            height=self.canvas_size + 120,
            resizable=True,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        self._is_dpg_initialized = True

    def render_topdown_canvas(
        self,
        grid,
        view_mode: str = "semantic",
    ) -> np.ndarray:
        """
        Renders a composite top-down 2.5D map using OpenCV.
        Draws resolution zones, observed cells with semantic colors or elevation,
        and concentric zone boundary rings with range labels.
        """
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        center = self.canvas_size // 2
        scale = (self.canvas_size / 2.0) / self.max_range  # pixels per meter

        # Background grid circles & dark backdrop
        cv2.circle(canvas, (center, center), int(self.max_range * scale), (25, 25, 25), -1)

        zm = grid.zoneManager() if hasattr(grid, "zoneManager") else grid.zone_manager()
        zones = zm.zones()

        palette = np.array([self.colormap.get_color_bgr(c) for c in range(6)], dtype=np.uint8)

        # Render cells from outermost zone (far) to innermost (immediate)
        for z_idx in reversed(range(len(zones))):
            z = zones[z_idx]
            cell_size = float(z.cell_size)
            offset_x = int(z.offset_x)
            offset_y = int(z.offset_y)

            # Extract map arrays directly from grid bindings
            count_map = grid.get_point_count_map(z_idx)
            sem_map = grid.get_semantic_map(z_idx)
            elev_map = grid.get_elevation_map(z_idx)

            ys, xs = np.nonzero(count_map > 0)
            if len(xs) == 0:
                continue

            # Vectorized coordinate computation
            wx = (xs - offset_x + 0.5) * cell_size
            wy = (ys - offset_y + 0.5) * cell_size
            px = np.int32(center + wx * scale)
            py = np.int32(center - wy * scale)

            valid = (px >= 0) & (px < self.canvas_size) & (py >= 0) & (py < self.canvas_size)
            if not np.any(valid):
                continue

            px_v = px[valid]
            py_v = py[valid]
            ys_v = ys[valid]
            xs_v = xs[valid]

            if view_mode == "elevation":
                z_vals = elev_map[ys_v, xs_v]
                norm_z = np.clip((z_vals + 2.0) / 6.0, 0.0, 1.0)
                b = (255 * (1.0 - norm_z)).astype(np.uint8)
                g = (255 * norm_z).astype(np.uint8)
                r = (255 * (1.0 - np.abs(norm_z - 0.5) * 2)).astype(np.uint8)
                colors = np.column_stack([b, g, r])
            else:
                cids = np.clip(sem_map[ys_v, xs_v], 0, 5)
                colors = palette[cids]

            pixel_radius = max(1, int(cell_size * scale * 0.8))
            if pixel_radius <= 1:
                canvas[py_v, px_v] = colors
            elif pixel_radius == 2:
                canvas[py_v, px_v] = colors
                canvas[np.clip(py_v + 1, 0, self.canvas_size - 1), px_v] = colors
                canvas[py_v, np.clip(px_v + 1, 0, self.canvas_size - 1)] = colors
                canvas[np.clip(py_v + 1, 0, self.canvas_size - 1), np.clip(px_v + 1, 0, self.canvas_size - 1)] = colors
            else:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        canvas[np.clip(py_v + dy, 0, self.canvas_size - 1),
                               np.clip(px_v + dx, 0, self.canvas_size - 1)] = colors



        # Draw zone boundary rings & annotations
        ring_colors = [(200, 200, 50), (255, 150, 0), (100, 255, 100), (0, 200, 255)]
        for i, z in enumerate(zones):
            r_pix = int(float(z.r_max) * scale)
            color = ring_colors[i % len(ring_colors)]
            cv2.circle(canvas, (center, center), r_pix, color, 1, cv2.LINE_AA)

            # Label on cardinal axis
            label = f"{z.name}: {z.r_max:.0f}m ({int(z.cell_size*100)}cm)"
            cv2.putText(
                canvas,
                label,
                (center + 5, center - r_pix + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )

        # Crosshairs and origin
        cv2.line(canvas, (center - 8, center), (center + 8, center), (255, 255, 255), 1)
        cv2.line(canvas, (center, center - 8), (center, center + 8), (255, 255, 255), 1)
        cv2.circle(canvas, (center, center), 3, (0, 0, 255), -1)  # Red sensor origin

        return canvas

    def compute_memory_statistics(self, grid) -> Dict[str, float]:
        """Compute foveated vs uniform 2.5D vs 3D voxel grid memory footprint."""
        total_foveated_cells = grid.totalCells() if hasattr(grid, "totalCells") else grid.total_cells()
        cell_bytes = 24  # sizeof(Cell)

        foveated_mb = (total_foveated_cells * cell_bytes) / (1024.0 * 1024.0)

        # Uniform 5cm 2.5D grid over 100m radius [-100, 100]: (200 / 0.05)^2 = 16,000,000 cells
        uniform_5cm_cells = (2.0 * self.max_range / 0.05) ** 2
        uniform_mb = (uniform_5cm_cells * cell_bytes) / (1024.0 * 1024.0)

        # Uniform 3D Voxel Grid: 4000 x 4000 x 120 (for 6m height span at 5cm): 1.92 x 10^9 voxels
        voxel_3d_cells = uniform_5cm_cells * (6.0 / 0.05)
        voxel_3d_mb = (voxel_3d_cells * 4) / (1024.0 * 1024.0)  # 4 bytes per voxel

        savings_pct = (1.0 - (total_foveated_cells / uniform_5cm_cells)) * 100.0

        return {
            "foveated_cells": float(total_foveated_cells),
            "foveated_mb": foveated_mb,
            "uniform_cells": float(uniform_5cm_cells),
            "uniform_mb": uniform_mb,
            "voxel_3d_cells": float(voxel_3d_cells),
            "voxel_3d_mb": voxel_3d_mb,
            "savings_pct": savings_pct,
        }

    def update_frame(
        self,
        grid,
        latency_ms: float = 12.0,
        view_mode: str = "semantic",
    ) -> np.ndarray:
        """
        Process a new grid frame, refresh telemetry, update Dear ImGui UI, and return canvas.
        """
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now

        fps = 1.0 / max(dt, 1e-4)
        self.fps_history.append(fps)
        self.latency_history.append(latency_ms)
        self.current_fps = float(np.mean(self.fps_history))
        self.current_latency_ms = latency_ms

        # 1. Render OpenCV 2.5D map canvas
        canvas_bgr = self.render_topdown_canvas(grid, view_mode=view_mode)
        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

        # 2. Compute memory statistics
        self.memory_stats = self.compute_memory_statistics(grid)

        # 3. Aggregate per-class cell counts
        zm = grid.zoneManager() if hasattr(grid, "zoneManager") else grid.zone_manager()
        total_counts = np.zeros(6, dtype=int)
        for z_idx in range(len(zm.zones())):
            count_map = grid.get_point_count_map(z_idx)
            sem_map = grid.get_semantic_map(z_idx)
            valid = count_map > 0
            if np.any(valid):
                cids = np.clip(sem_map[valid], 0, 5)
                total_counts += np.bincount(cids, minlength=6)
        self.class_counts = {c: int(total_counts[c]) for c in range(6)}


        # 4. Update DearPyGui / OpenGL widgets if GUI is active
        if self._is_dpg_initialized and dpg.is_dearpygui_running():
            # Update texture buffer (RGBA float normalized [0, 1])
            rgba = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
            dpg.set_value("texture_2d_map", rgba.ravel())


            # Update telemetry text
            dpg.set_value("txt_fps", f"FPS: {self.current_fps:.1f} (target: >= 10 Hz)")
            dpg.set_value("txt_latency", f"Latency: {latency_ms:.1f} ms (p95: {np.percentile(list(self.latency_history), 95):.1f} ms)")

            # Update latency plot
            frames = list(range(len(self.latency_history)))
            dpg.set_value("series_latency", [frames, list(self.latency_history)])

            # Update memory text
            dpg.set_value("txt_mem_foveated", f"Foveated (2.5D): {self.memory_stats['foveated_mb']:.1f} MB ({int(self.memory_stats['foveated_cells']):,} cells)")
            dpg.set_value("txt_mem_uniform", f"Uniform (5cm):  {self.memory_stats['uniform_mb']:.1f} MB ({int(self.memory_stats['uniform_cells']):,} cells)")
            dpg.set_value("txt_mem_3d", f"3D Voxel Grid:  {self.memory_stats['voxel_3d_mb']:.1f} MB ({int(self.memory_stats['voxel_3d_cells']):,} voxels)")
            dpg.set_value("txt_mem_savings", f"Memory Reduction: {self.memory_stats['savings_pct']:.2f}%")

            # Update class counts
            for cid in range(6):
                cname = self.colormap.get_class_name(cid)
                dpg.set_value(f"txt_class_{cid}", f"{cname}: {self.class_counts.get(cid, 0):,} cells")

            # Render DearPyGui frame
            dpg.render_dearpygui_frame()

        return canvas_bgr

    def close(self) -> None:
        """Clean up DearPyGui context."""
        if self._is_dpg_initialized:
            dpg.destroy_context()
            self._is_dpg_initialized = False


# Interactive Real-Time Loop & Demonstration
def run_interactive_dashboard(headless: bool = False, canvas_size: int = 600, fps_target: int = 20):
    """Run an interactive real-time dashboard with dynamic Lidar scene animation."""
    from tests.test_grid_engine import build_configured_zone_manager
    from demo.run_demo import DynamicDrivingSimulator

    print("[Dashboard] Initializing DRDO PS 26053 Tactical HUD...")
    zm = build_configured_zone_manager()
    grid = grid_py.FoveatedGrid(zm)
    simulator = DynamicDrivingSimulator(num_points_per_frame=20000)

    dashboard = GridDashboard(canvas_size=canvas_size, headless=headless)


    if dashboard.headless:
        print("[Dashboard] Running in headless mode (no display detected or --headless specified).")
        pts_raw, _ = simulator.get_next_frame()
        # Create random classes for simulation
        rng = np.random.default_rng(42)
        classes = rng.choice([0, 1, 2, 3, 4], size=len(pts_raw), p=[0.45, 0.25, 0.15, 0.10, 0.05])
        confs = rng.uniform(0.7, 1.0, size=len(pts_raw))
        classified_pts = np.column_stack([pts_raw[:, :3], classes, confs]).astype(np.float32)

        grid.clear()
        grid.projectPoints(classified_pts)
        grid.finalizeCells()

        canvas = dashboard.update_frame(grid, latency_ms=14.2)
        out_path = PROJECT_ROOT / "demo" / "dashboard_snapshot.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        print(f"[Dashboard] Snapshot saved to: {out_path}")
        print(f"[Dashboard] Memory stats: {dashboard.memory_stats}")
        dashboard.close()
        return

    print("[Dashboard] Window opened! Showing real-time Foveated 2.5D Lidar map.")
    print("[Dashboard] Close the window or press Ctrl+C to exit.")

    frame_interval = 1.0 / max(1, fps_target)
    rng = np.random.default_rng(101)

    try:
        while dpg.is_dearpygui_running():
            t_start = time.perf_counter()

            # Generate next simulated dynamic lidar scan
            pts_raw, _ = simulator.get_next_frame()
            n = len(pts_raw)
            # Synthetic classification based on geometry & actors
            classes = np.zeros(n, dtype=np.float32)
            # Mark actors based on height/position
            z = pts_raw[:, 2]
            y = pts_raw[:, 1]
            x = pts_raw[:, 0]
            # Sidewalks / non-drivable
            classes[(np.abs(x) > 4.5)] = 1.0
            # Obstacles / buildings
            classes[(np.abs(x) > 10.0) & (z > -0.5)] = 2.0
            # Dynamic vehicles
            classes[(y > 18.0) & (y < 28.0) & (np.abs(x) < 3.0) & (z > -1.2)] = 3.0
            classes[(y > 38.0) & (y < 46.0) & (np.abs(x) < 3.5) & (z > -1.2)] = 3.0
            # Pedestrians
            classes[(y > 12.0) & (y < 16.0) & (x > -4.5) & (x < 1.0) & (z > -1.4)] = 4.0

            confs = rng.uniform(0.75, 0.99, size=n).astype(np.float32)
            classified_pts = np.column_stack([pts_raw[:, :3], classes, confs]).astype(np.float32)

            t_proj_0 = time.perf_counter()
            grid.clear()
            grid.projectPoints(classified_pts)
            grid.finalizeCells()
            latency_ms = (time.perf_counter() - t_proj_0) * 1000.0 + 12.5  # include simulated model latency

            dashboard.update_frame(grid, latency_ms=latency_ms)

            # Cap frame rate
            elapsed = time.perf_counter() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Dashboard] Exiting...")
    finally:
        dashboard.close()
        print("[Dashboard] Closed successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Dear ImGui Foveated 2.5D Lidar Dashboard")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless and save snapshot")
    parser.add_argument("--size", type=int, default=600, help="Canvas dimension in pixels (default: 600)")
    parser.add_argument("--fps", type=int, default=20, help="Target update FPS (default: 20)")
    args = parser.parse_args()

    run_interactive_dashboard(headless=args.headless, canvas_size=args.size, fps_target=args.fps)

