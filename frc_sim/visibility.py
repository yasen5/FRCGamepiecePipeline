"""Conservative line-of-sight checks for the fixed left COS camera."""
from dataclasses import dataclass
from math import radians, tan

import cv2
import numpy as np

from .camera import Matrix, robot_camera_pose
from .model import Vec3, default_calibration

Bounds3D = tuple[Vec3, Vec3]
_HORIZONTAL_SLOPE = tan(radians(70 / 2))
_VERTICAL_SLOPE = tan(radians(47.3 / 2))


@dataclass(frozen=True)
class SightTarget:
    points: Matrix
    bounds: Bounds3D


def _blocked(origin: Matrix, target: Matrix, bounds: Matrix) -> bool:
    # A segment intersects a box if its per-axis entry and exit intervals overlap.
    low, high = bounds[:, 0], bounds[:, 1]
    near = np.all((low <= np.maximum(origin, target)) & (high >= np.minimum(origin, target)), axis=1)
    if not np.any(near):
        return False
    direction = target - origin
    moving = direction != 0
    # The broad phase already checked axes on which the segment is stationary.
    first = (low[near][:, moving] - origin[moving]) / direction[moving]
    last = (high[near][:, moving] - origin[moving]) / direction[moving]
    entry = np.maximum(0, np.minimum(first, last).max(axis=1, initial=-np.inf))
    exit_ = np.minimum(1, np.maximum(first, last).min(axis=1, initial=np.inf))
    return bool(np.any(entry <= exit_))


class LeftCameraVisibility:
    def __init__(self, obstacles: list[Bounds3D], targets: list[SightTarget]) -> None:
        self.calibration = default_calibration()
        c = self.calibration
        i = c.intrinsics
        self.intrinsics = np.array([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1]], dtype=np.float64)
        self.distortion = np.array([i.k1, i.k2, i.p1, i.p2, i.k3])
        self.zero = np.zeros(3)
        self.targets = targets
        self.points = np.concatenate([target.points for target in targets]) if targets else np.empty((0, 3))
        self.indices = np.repeat(np.arange(len(targets)), np.array([len(target.points) for target in targets], dtype=np.intp))
        self.field = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2, 3)
        self.pieces = np.asarray([target.bounds for target in targets], dtype=np.float64).reshape(-1, 2, 3)

    def meets_minimum(self, pose: Vec3, minimum: int) -> bool:
        if minimum <= 0:
            return True
        camera = robot_camera_pose(pose, self.calibration)
        origin = camera[:3, 3]
        forward, left, up = ((self.points - origin) @ camera[:3, :3]).T
        candidates = np.flatnonzero((forward > .02) & (forward < 100)
                                    & (np.abs(left) <= forward * _HORIZONTAL_SLOPE)
                                    & (np.abs(up) <= forward * _VERTICAL_SLOPE))
        if np.unique(self.indices[candidates]).size < minimum:
            return False
        optical = np.column_stack((-left[candidates], -up[candidates], forward[candidates]))
        pixels, _ = cv2.projectPoints(optical, self.zero, self.zero, self.intrinsics, self.distortion)
        u, v = pixels.reshape(-1, 2).T
        c = self.calibration
        in_frame = candidates[(u >= 0) & (u < c.width) & (v >= 0) & (v < c.height)]
        layout_indices = self.indices[in_frame]
        visible = 0
        for index in np.unique(layout_indices):
            others = self.pieces[np.arange(len(self.targets)) != index]
            for point in self.points[in_frame[layout_indices == index]]:
                if not _blocked(origin, point, self.field) and not _blocked(origin, point, others):
                    visible += 1
                    if visible >= minimum:
                        return True
                    break
        return False
