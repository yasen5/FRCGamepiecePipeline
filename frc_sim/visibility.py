"""Conservative line-of-sight checks for the fixed left COS camera."""
from dataclasses import dataclass
from math import hypot

import numpy as np

from .camera import ImageProjector, Matrix, robot_camera_pose
from .model import Vec3, default_calibration

Bounds3D = tuple[Vec3, Vec3]


@dataclass(frozen=True)
class SightTarget:
    points: Matrix
    bounds: Bounds3D
    center: Vec3


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
    def __init__(self, obstacles: list[Bounds3D], targets: list[SightTarget], *,
                 max_distance: float | None = None) -> None:
        self.calibration = default_calibration()
        self.projector = ImageProjector(self.calibration, distorted=True)
        self.max_distance = max_distance
        self.centers = [target.center for target in targets]
        self.required = [len(target.points) // 2 + 1 for target in targets]
        self.points = np.concatenate([target.points for target in targets]) if targets else np.empty((0, 3))
        self.indices = np.repeat(np.arange(len(targets)), np.array([len(target.points) for target in targets], dtype=np.intp))
        self.field = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2, 3)
        self.pieces = np.asarray([target.bounds for target in targets], dtype=np.float64).reshape(-1, 2, 3)

    def meets_minimum(self, pose: Vec3, minimum: int) -> bool:
        if minimum <= 0:
            return True
        camera = robot_camera_pose(pose, self.calibration)
        origin = camera[:3, 3]
        in_frame, _ = self.projector.project(self.points, camera)
        layout_indices = self.indices[in_frame]
        visible = 0
        for index in np.unique(layout_indices):
            if self.max_distance is not None:
                x, y, _ = self.centers[index]
                if hypot(x - origin[0], y - origin[1]) > self.max_distance:
                    continue
            candidates = in_frame[layout_indices == index]
            required = self.required[index]
            if candidates.size < required:
                continue
            others = self.pieces[np.arange(len(self.required)) != index]
            seen = 0
            for point in self.points[candidates]:
                if not _blocked(origin, point, self.field) and not _blocked(origin, point, others):
                    seen += 1
                    if seen == required:
                        visible += 1
                        if visible >= minimum:
                            return True
                        break
        return False
