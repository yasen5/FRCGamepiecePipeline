"""Canonical landmarks and calibrated frame annotations."""
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from .camera import ImageProjector, Matrix, camera_pose, piece_transform
from .interaction import vertices
from .model import Keypoint, Record, Scene, Vec3, stock_gamepiece_paths

PieceKind = Literal['cone', 'cube']
Landmarks = Annotated[list[Keypoint], Field(min_length=1)]


class LayoutKeypoints(Record):
    cone: Landmarks | None = None
    cube: Landmarks | None = None

    def for_kind(self, kind: PieceKind) -> list[Keypoint]:
        points = self.cone if kind == 'cone' else self.cube
        if points is None:
            raise ValueError(f'Missing keypoint definitions for {kind}')
        return points


class ImagePoint(Record):
    class_id: int
    piece_index: int
    x: float
    y: float


class ImageAnnotation(Record):
    image: str
    width: int
    height: int
    points: list[ImagePoint]


class KeypointManifest(Record):
    version: Literal[1] = 1
    classes: list[str]
    images: list[ImageAnnotation]

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + '\n')


def mesh_center(points: Matrix) -> Vec3:
    x, y, z = (points.min(axis=0) + points.max(axis=0)) / 2
    return float(x), float(y), float(z)


def world_points(points: list[Vec3], position: Vec3, rotation: Vec3,
                 import_rotation: Vec3, scale: float = 1.) -> Matrix:
    pose = piece_transform(position, rotation, import_rotation)
    local = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (local * scale) @ pose[:3, :3].T + pose[:3, 3]


@dataclass(frozen=True)
class SceneLandmarks:
    """World points for a fixed set of placed pieces across camera poses."""
    points: Matrix
    labels: list[tuple[int, str]]

    @classmethod
    def from_scene(cls, scene: Scene, named: bool) -> 'SceneLandmarks':
        stock = stock_gamepiece_paths(scene.field_dir)
        arrays: list[Matrix] = []
        labels: list[tuple[int, str]] = []
        for index, piece in enumerate(scene.pieces):
            path = Path(piece.path)
            if piece.gamepiece is None and path.resolve() not in stock:
                continue
            if named:
                points = piece.keypoints
                if not points:
                    raise ValueError(f'Missing keypoint definitions for scene piece {index} ({piece.name})')
            else:
                center = mesh_center(vertices(str(path.resolve()), path.stat().st_mtime_ns))
                points = [Keypoint(name='center', position=center)]
            arrays.append(world_points([point.position for point in points], piece.position,
                                       piece.rotation, piece.import_rotation, piece.scale))
            labels.extend((index, point.name) for point in points)
        return cls(np.concatenate(arrays) if arrays else np.empty((0, 3), dtype=np.float64), labels)

    def annotate(self, scene: Scene, projector: ImageProjector, image: str,
                 class_ids: dict[str, int]) -> ImageAnnotation:
        indices, pixels = projector.project(self.points, camera_pose(scene))
        points: list[ImagePoint] = []
        for index, (x, y) in zip(indices, pixels):
            piece_index, name = self.labels[index]
            points.append(ImagePoint(class_id=class_ids[name], piece_index=piece_index, x=float(x), y=float(y)))
        return ImageAnnotation(image=image, width=projector.width, height=projector.height, points=points)
