"""Camera geometry and cached OpenCV Brown–Conrady inverse sampling."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from numpy.typing import NDArray

from .model import Calibration, Scene, Vec3

Matrix = NDArray[np.float64]


def transform(position: Vec3, angles: Vec3) -> Matrix:
    """NWU local-to-parent transform; Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = np.deg2rad(angles)
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    result = np.eye(4)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                       [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                       [-sp, cp*sr, cp*cr]]
    result[:3, 3] = position
    return result


def camera_pose(scene: Scene) -> Matrix:
    if scene.calibration is None:
        raise ValueError("Load camera calibration before exporting")
    e = scene.calibration.extrinsics
    x, y, heading = scene.robot_xy_heading
    return transform((x, y, 0), (0, 0, heading)) @ transform(
        (e.translation_x, e.translation_y, e.translation_z),
        (e.rotation_x, e.rotation_y, e.rotation_z))


def inspection_pose(scene: Scene) -> Matrix:
    yaw, pitch = np.deg2rad([scene.inspect_yaw, scene.inspect_pitch])
    offset = scene.inspect_distance * np.array([np.cos(pitch)*np.cos(yaw),
                                               np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
    target = np.array(scene.inspect_target)
    forward = -offset / np.linalg.norm(offset)
    left = np.cross([0, 0, 1], forward)
    left /= np.linalg.norm(left)
    result = np.eye(4)
    result[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
    result[:3, 3] = target + offset
    return result


@dataclass(frozen=True)
class Projection:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    map_x: NDArray[np.float32] | None = None
    map_y: NDArray[np.float32] | None = None


@lru_cache(maxsize=8)
def projection(calibration_json: str, preview: bool, distorted: bool) -> Projection:
    c = Calibration.model_validate_json(calibration_json)
    scale = min(1.0, 960 / c.width, 540 / c.height) if preview else 1.0
    w, h = round(c.width*scale), round(c.height*scale)
    sx, sy = w/c.width, h/c.height
    i = c.intrinsics
    fx, fy, cx, cy = i.fx*sx, i.fy*sy, i.cx*sx, i.cy*sy
    if not distorted:
        return Projection(w, h, fx, fy, cx, cy)
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    d = np.array([i.k1, i.k2, i.p1, i.p2, i.k3])
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack([xx, yy], axis=-1).astype(np.float64).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pixels, k, d, P=k).reshape(h, w, 2)
    if not np.isfinite(ideal).all():
        raise ValueError("Calibration produces invalid lens mapping")
    low = np.floor(ideal.min(axis=(0, 1))).astype(int) - 2
    high = np.ceil(ideal.max(axis=(0, 1))).astype(int) + 2
    rw, rh = int(high[0]-low[0]+1), int(high[1]-low[1]+1)
    if rw > 16384 or rh > 16384 or rw*rh > 40_000_000:
        raise ValueError("Lens distortion requires excessive overscan")
    return Projection(rw, rh, fx, fy, cx-low[0], cy-low[1],
                      (ideal[..., 0]-low[0]).astype(np.float32),
                      (ideal[..., 1]-low[1]).astype(np.float32))
