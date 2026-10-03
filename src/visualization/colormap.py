# =============================================================================
# colormap.py — Semantic Class Color Palette & Grid Colorization Utilities
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import yaml

# Default color mapping matching config/grid_params.yaml
DEFAULT_SEMANTIC_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 200, 0),       # drivable_surface: green
    1: (139, 90, 43),     # non_drivable_terrain: brown
    2: (128, 128, 128),   # static_obstacle: grey
    3: (0, 100, 255),     # dynamic_vehicle: blue
    4: (255, 0, 0),       # dynamic_pedestrian: red
    5: (50, 50, 50),      # unknown_noise: dark grey
}

DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "drivable_surface",
    1: "non_drivable_terrain",
    2: "static_obstacle",
    3: "dynamic_vehicle",
    4: "dynamic_pedestrian",
    5: "unknown_noise",
}


class Colormap:
    """Manages semantic class color conversions between RGB, RGBA, BGR, and normalized floats."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.colors: Dict[int, Tuple[int, int, int]] = dict(DEFAULT_SEMANTIC_COLORS)
        self.names: Dict[int, str] = dict(DEFAULT_CLASS_NAMES)

        if config_path is not None:
            self.load_from_yaml(config_path)

        # Precompute fast color lookup table (256, 3) for vectorized numpy coloring
        self._lut_rgb = np.zeros((256, 3), dtype=np.uint8)
        self._lut_bgr = np.zeros((256, 3), dtype=np.uint8)
        for cid, (r, g, b) in self.colors.items():
            if 0 <= cid < 256:
                self._lut_rgb[cid] = [r, g, b]
                self._lut_bgr[cid] = [b, g, r]

    def load_from_yaml(self, config_path: Union[str, Path]) -> None:
        """Load color definitions from grid_params.yaml."""
        p = Path(config_path)
        if not p.is_file():
            return
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)

        if "semantic_classes" in cfg:
            for k, val in cfg["semantic_classes"].items():
                cid = int(k)
                if "color" in val:
                    self.colors[cid] = tuple(val["color"])
                if "name" in val:
                    self.names[cid] = str(val["name"])

    def get_color_rgb(self, class_id: int, normalized: bool = False) -> Tuple[float, float, float]:
        """Return (R, G, B) as uint8 [0, 255] or float [0.0, 1.0]."""
        c = self.colors.get(int(class_id), (50, 50, 50))
        if normalized:
            return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
        return c

    def get_color_rgba(self, class_id: int, alpha: float = 1.0, normalized: bool = False) -> Tuple[float, float, float, float]:
        """Return (R, G, B, A) as uint8 [0, 255] or float [0.0, 1.0]."""
        r, g, b = self.get_color_rgb(class_id, normalized=normalized)
        a = float(alpha) if normalized else float(alpha * 255.0)
        return (r, g, b, a)

    def get_color_bgr(self, class_id: int) -> Tuple[int, int, int]:
        """Return (B, G, R) for OpenCV visualization."""
        r, g, b = self.get_color_rgb(class_id, normalized=False)
        return (int(b), int(g), int(r))

    def get_class_name(self, class_id: int) -> str:
        """Return human-readable class name."""
        return self.names.get(int(class_id), f"class_{class_id}")

    def colorize_semantic_grid(self, grid_2d: np.ndarray, bgr: bool = False) -> np.ndarray:
        """
        Colorize a 2D grid of class IDs (H, W) into an RGB or BGR image (H, W, 3).
        Vectorized with a 256-element lookup table.
        """
        arr = np.asarray(grid_2d, dtype=np.uint8)
        lut = self._lut_bgr if bgr else self._lut_rgb
        return lut[arr]

    def colorize_elevation_grid(
        self,
        elevation_2d: np.ndarray,
        vmin: float = -2.0,
        vmax: float = 4.0,
        unobserved_color: Tuple[int, int, int] = (20, 20, 20),
    ) -> np.ndarray:
        """
        Colorize a 2D elevation grid into a Turbo/Jet-style RGB heatmap.
        Unobserved cells (NaNs or extreme values) are masked to unobserved_color.
        """
        arr = np.asarray(elevation_2d, dtype=np.float32)
        valid_mask = np.isfinite(arr) & (arr > -100.0) & (arr < 100.0)

        # Normalize valid values to [0, 1]
        safe_arr = np.nan_to_num(arr, nan=vmin)
        norm = np.clip((safe_arr - vmin) / max(vmax - vmin, 1e-4), 0.0, 1.0)

        # Fast pseudo-jet / turbo colormap approximation
        # R = max(0, 1.5 - |4*norm - 3|)
        # G = max(0, 1.5 - |4*norm - 2|)
        # B = max(0, 1.5 - |4*norm - 1|)
        r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)

        rgb = np.stack([r, g, b], axis=-1)
        rgb = (rgb * 255.0).astype(np.uint8)

        # Apply unobserved color
        rgb[~valid_mask] = unobserved_color
        return rgb


# Global default instance
default_colormap = Colormap()
