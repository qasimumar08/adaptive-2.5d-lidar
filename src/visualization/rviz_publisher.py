# =============================================================================
# rviz_publisher.py — Converts Foveated 2.5D Grid Cells to RViz2 MarkerArray
# PS 26053: Adaptive Variable Resolution 2.5D Lidar Mapping
# =============================================================================

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.colormap import Colormap, default_colormap

try:
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA, Header
    from visualization_msgs.msg import Marker, MarkerArray
    _m = Marker()
    _m.points.append(Point(x=0.0, y=0.0, z=0.0))
    HAS_ROS2_MSGS = True
except Exception:
    class Point:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class ColorRGBA:
        def __init__(self, r=1.0, g=1.0, b=1.0, a=1.0):
            self.r, self.g, self.b, self.a = float(r), float(g), float(b), float(a)

    class Header:
        def __init__(self, frame_id="lidar_link", stamp=None):
            self.frame_id = frame_id
            self.stamp = stamp

    class _Pose:
        def __init__(self):
            self.position = Point(0.0, 0.0, 0.0)
            self.orientation = type("Quat", (), {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})()

    class _Scale:
        def __init__(self, x=1.0, y=1.0, z=1.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class Marker:
        ARROW = 0
        CUBE = 1
        SPHERE = 2
        CYLINDER = 3
        LINE_STRIP = 4
        LINE_LIST = 5
        CUBE_LIST = 6
        SPHERE_LIST = 7
        POINTS = 8
        TEXT_VIEW_FACING = 9
        MESH_RESOURCE = 10
        TRIANGLE_LIST = 11

        ADD = 0
        MODIFY = 0
        DELETE = 2
        DELETEALL = 3

        def __init__(self):
            self.header = Header()
            self.ns = ""
            self.id = 0
            self.type = Marker.CUBE_LIST
            self.action = Marker.ADD
            self.pose = _Pose()
            self.scale = _Scale()
            self.color = ColorRGBA()
            self.points = []
            self.colors = []

    class MarkerArray:
        def __init__(self):
            self.markers = []

    HAS_ROS2_MSGS = True


class RVizGridVisualizer:
    """
    Converts 2.5D FoveatedGrid cells into high-performance RViz2 MarkerArray messages.
    Uses CUBE_LIST markers per zone (togglable in RViz) with exact zone cell sizes and colors.
    """

    def __init__(
        self,
        frame_id: str = "lidar_link",
        colormap: Optional[Colormap] = None,
        min_points_threshold: int = 1,
        default_cube_height: float = 0.10,
    ):
        self.frame_id = frame_id
        self.colormap = colormap or default_colormap
        self.min_points_threshold = min_points_threshold
        self.default_cube_height = default_cube_height

    def grid_to_marker_array(
        self,
        grid,
        stamp=None,
        include_free_surface: bool = True,
        include_zone_rings: bool = True,
    ):
        """
        Convert a FoveatedGrid instance into a visualization_msgs.msg.MarkerArray.

        Args:
            grid: grid_py.FoveatedGrid instance
            stamp: builtin_interfaces.msg.Time or None
            include_free_surface: if True, render drivable surface (class 0) cells as thin ground tiles
            include_zone_rings: if True, add circular line markers indicating zone boundary radii

        Returns:
            visualization_msgs.msg.MarkerArray or dict (if ROS2 msgs unavailable)
        """
        if not HAS_ROS2_MSGS:
            return self._to_fallback_dict(grid)

        marker_array = MarkerArray()
        zm = grid.zoneManager() if hasattr(grid, "zoneManager") else grid.zone_manager()
        zones = zm.zones()

        for z_idx, z in enumerate(zones):
            cell_size = float(z.cell_size)
            width = int(z.grid_width)
            height = int(z.grid_height)
            offset_x = int(z.offset_x)
            offset_y = int(z.offset_y)

            # CUBE_LIST marker for this zone
            cube_marker = Marker()
            cube_marker.header.frame_id = self.frame_id
            if stamp is not None:
                cube_marker.header.stamp = stamp
            cube_marker.ns = f"foveated_zone_{z_idx}_{z.name}"
            cube_marker.id = z_idx
            cube_marker.type = Marker.CUBE_LIST
            cube_marker.action = Marker.ADD

            cube_marker.scale.x = cell_size * 0.95  # Slight gap for grid outline definition
            cube_marker.scale.y = cell_size * 0.95
            cube_marker.scale.z = self.default_cube_height

            cube_marker.pose.orientation.w = 1.0

            # Scan cells in this zone
            for gy in range(height):
                for gx in range(width):
                    cell = grid.getCell(z_idx, gx, gy)
                    if cell.point_count < self.min_points_threshold:
                        continue

                    # Filter out drivable surface if requested
                    if not include_free_surface and cell.semantic_class == 0:
                        continue

                    # Center coordinate of cell in lidar_link frame
                    x = (gx - offset_x + 0.5) * cell_size
                    y = (gy - offset_y + 0.5) * cell_size
                    z_elev = float(cell.elevation_mean())

                    pt = Point()
                    pt.x = float(x)
                    pt.y = float(y)
                    pt.z = z_elev
                    cube_marker.points.append(pt)

                    # Color per semantic class
                    r, g, b = self.colormap.get_color_rgb(cell.semantic_class, normalized=True)
                    # Alpha: slightly transparent for drivable surface, opaque for obstacles
                    alpha = 0.6 if cell.semantic_class == 0 else 0.95
                    color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(alpha))
                    cube_marker.colors.append(color)

            # Only append if points were found in this zone
            if len(cube_marker.points) > 0:
                marker_array.markers.append(cube_marker)

        # Optional: Add boundary ring markers to clearly visualize resolution transitions
        if include_zone_rings:
            rings_marker = self._create_zone_rings_marker(zones, stamp)
            marker_array.markers.append(rings_marker)

        return marker_array

    def _create_zone_rings_marker(self, zones, stamp=None) -> "Marker":
        """Create LINE_LIST boundary rings showing radial zones (0-10m, 10-30m, 30-60m, 60-100m)."""
        ring = Marker()
        ring.header.frame_id = self.frame_id
        if stamp is not None:
            ring.header.stamp = stamp
        ring.ns = "zone_boundary_rings"
        ring.id = 999
        ring.type = Marker.LINE_LIST
        ring.action = Marker.ADD
        ring.scale.x = 0.08  # Line width in meters
        ring.pose.orientation.w = 1.0

        ring_color = ColorRGBA(r=1.0, g=1.0, b=0.2, a=0.8)  # Yellow rings
        ring.color = ring_color

        segments = 72  # 5-degree increments
        for z in zones:
            r = float(z.r_max)
            prev_pt = None
            first_pt = None
            for s in range(segments + 1):
                theta = (2.0 * math.pi * s) / segments
                curr_pt = Point(x=r * math.cos(theta), y=r * math.sin(theta), z=0.0)
                if prev_pt is not None:
                    ring.points.append(prev_pt)
                    ring.points.append(curr_pt)
                else:
                    first_pt = curr_pt
                prev_pt = curr_pt

        return ring

    def _to_fallback_dict(self, grid) -> dict:
        """Fallback serialization for headless or non-ROS environments."""
        zm = grid.zoneManager() if hasattr(grid, "zoneManager") else grid.zone_manager()
        data = {"frame_id": self.frame_id, "zones": []}
        for z_idx, z in enumerate(zm.zones()):
            occupied = []
            for gy in range(z.grid_height):
                for gx in range(z.grid_width):
                    c = grid.getCell(z_idx, gx, gy)
                    if c.point_count >= self.min_points_threshold:
                        occupied.append({
                            "gx": gx,
                            "gy": gy,
                            "mean_z": c.elevation_mean(),
                            "class": int(c.semantic_class),
                            "count": int(c.point_count),
                        })
            data["zones"].append({"name": z.name, "cells": occupied})
        return data
