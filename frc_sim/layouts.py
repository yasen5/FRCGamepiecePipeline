"""Generate compact JSON layouts in blue-origin field coordinates."""
from dataclasses import dataclass
from hashlib import sha256
from math import cos, isfinite, radians, sin
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PositiveFloat, ValidationError

from .camera import transform
from .field import FieldConfig
from .keypoints import LayoutKeypoints, PieceKind, mesh_center, world_points
from .model import DEFAULT_CONE_PATH, Record, Vec3, default_cone
from .visibility import Bounds3D, LeftCameraVisibility, SightTarget

BoundsXY = tuple[float, float, float, float]
MAX_ATTEMPTS = 1000
MAX_LAYOUT_ATTEMPTS = 20
CUBE_IMPORT_ROTATION: Vec3 = (90, 0, 0)


class GamepiecePose(Record):
    kind: PieceKind
    position: Vec3
    rotation: Vec3


class LayoutBatch(Record):
    version: Literal[1] = 1
    gamepieces: list[GamepiecePose]
    robot_poses: Annotated[list[Vec3], Field(min_length=1)]


class PieceFootprint(Record):
    height_m: float
    radius_m: PositiveFloat
    bounds: Bounds3D


class LayoutGeometry(Record):
    field_size_m: tuple[PositiveFloat, PositiveFloat]
    obstacles: list[BoundsXY]
    occluders: list[Bounds3D]
    cone_footprints: tuple[PieceFootprint, PieceFootprint]
    cube_footprints: tuple[PieceFootprint, PieceFootprint]
    cone_center: Vec3
    cube_center: Vec3


@dataclass(frozen=True)
class LayoutOptions:
    batch_size: int
    num_batches: int = 1
    piece_count_mean: float = 8
    piece_count_std: float = 3
    robot_length: float = .8
    robot_width: float = .8
    seed: int | None = None
    min_visible: int = 0

    def __post_init__(self) -> None:
        if self.min_visible < 0:
            raise ValueError('Minimum visible gamepieces must be nonnegative')
        if self.batch_size < 1 or self.num_batches < 1:
            raise ValueError('Batch size and number of batches must be positive')
        for value in (self.robot_length, self.robot_width):
            if not isfinite(value) or value <= 0:
                raise ValueError('Robot dimensions must be finite and positive')
        for value in (self.piece_count_mean, self.piece_count_std):
            if not isfinite(value) or value < 0:
                raise ValueError('Piece count mean and standard deviation must be finite and nonnegative')
        if self.piece_count_std == 0 and round(self.piece_count_mean) < self.min_visible:
            raise ValueError('The fixed gamepiece count is smaller than the minimum visibility requirement')


def _preprocess(config_path: Path, field_path: Path, cone_path: Path, cube_path: Path) -> LayoutGeometry:
    import trimesh

    from .interaction import vertices

    config = FieldConfig.model_validate_json(config_path.read_bytes())
    width, height = config.size_m
    field_pose = config.world_transform()
    model = trimesh.load_scene(field_path)
    nodes = list(model.graph.nodes_geometry)
    staged = config.staged_nodes(nodes)
    obstacles: list[BoundsXY] = []
    occluders: list[Bounds3D] = []
    for node in nodes:
        if node in staged:
            continue
        node_pose, name = model.graph[node]
        mesh = model.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f'Expected triangle mesh for field node {node}')
        pose = field_pose @ node_pose
        world = np.asarray(mesh.vertices, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
        low, high = world.min(axis=0), world.max(axis=0)
        # Ignore the carpet and thin floor markings at the nominal ground plane.
        if high[2] <= .0033:
            continue
        occluders.append((_vec3(low), _vec3(high)))
        if high[0] < 0 or high[1] < 0 or low[0] > width or low[1] > height:
            continue
        obstacles.append((float(low[0]), float(low[1]), float(high[0]), float(high[1])))

    def footprints(points: NDArray[np.float64], import_rotation: Vec3) -> tuple[PieceFootprint, PieceFootprint]:
        correction = transform((0, 0, 0), import_rotation)
        results: list[PieceFootprint] = []
        for pitch in (0., 90.):
            pose = transform((0, 0, 0), (0, pitch, 0)) @ correction
            rotated = points @ pose[:3, :3].T
            results.append(PieceFootprint(height_m=-float(rotated[:, 2].min()),
                                          radius_m=float(np.linalg.norm(rotated[:, :2], axis=1).max()),
                                          bounds=(_vec3(rotated.min(axis=0)), _vec3(rotated.max(axis=0)))))
        return results[0], results[1]

    cone = vertices(str(cone_path), cone_path.stat().st_mtime_ns)
    cube = vertices(str(cube_path), cube_path.stat().st_mtime_ns)
    return LayoutGeometry(field_size_m=(width, height), obstacles=obstacles, occluders=occluders,
                          cone_footprints=footprints(cone, default_cone().import_rotation),
                          cube_footprints=footprints(cube, CUBE_IMPORT_ROTATION),
                          cone_center=mesh_center(cone), cube_center=mesh_center(cube))


def load_geometry(cache_dir: Path, field_dir: Path = Path('assets/field'),
                  cone_path: Path = Path(DEFAULT_CONE_PATH)) -> LayoutGeometry:
    config = field_dir / 'config.json'
    field = field_dir / 'model.glb'
    cube = field_dir / 'model_1.glb'
    key = sha256(b'layout-geometry-v3')
    for asset in (config, field, cone_path, cube):
        info = asset.stat()
        key.update(repr((str(asset.resolve()), info.st_size, info.st_mtime_ns, info.st_ctime_ns)).encode())
    cache = cache_dir / f'layout-geometry-{key.hexdigest()}.json'
    try:
        return LayoutGeometry.model_validate_json(cache.read_bytes())
    except (FileNotFoundError, ValidationError):
        geometry = _preprocess(config, field, cone_path, cube)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache_dir) as temporary:
            staged = Path(temporary) / cache.name
            staged.write_text(geometry.model_dump_json())
            staged.replace(cache)
        return geometry


class _Field:
    def __init__(self, geometry: LayoutGeometry) -> None:
        self.width, self.height = geometry.field_size_m
        boxes = np.asarray(geometry.obstacles, dtype=np.float64).reshape(-1, 4)
        self.low_x, self.low_y, self.high_x, self.high_y = boxes.T

    def contains(self, x: float, y: float, extent_x: float, extent_y: float) -> bool:
        return extent_x < x < self.width - extent_x and extent_y < y < self.height - extent_y

    def nearby(self, x: float, y: float, extent_x: float, extent_y: float) -> NDArray[np.bool_]:
        return ((self.low_x <= x + extent_x) & (self.high_x >= x - extent_x)
                & (self.low_y <= y + extent_y) & (self.high_y >= y - extent_y))

    def circle_is_clear(self, x: float, y: float, radius: float) -> bool:
        if not self.contains(x, y, radius, radius):
            return False
        near = self.nearby(x, y, radius, radius)
        dx = np.maximum(self.low_x[near] - x, x - self.high_x[near]).clip(min=0)
        dy = np.maximum(self.low_y[near] - y, y - self.high_y[near]).clip(min=0)
        return not np.any(dx * dx + dy * dy <= radius * radius)

    def robot_is_clear(self, pose: Vec3, half_length: float, half_width: float,
                       pieces: list[tuple[float, float, float]]) -> bool:
        x, y, yaw = pose
        c, s = cos(radians(yaw)), sin(radians(yaw))
        extent_x = abs(c) * half_length + abs(s) * half_width
        extent_y = abs(s) * half_length + abs(c) * half_width
        if not self.contains(x, y, extent_x, extent_y):
            return False

        # The broad phase tests the field axes. The remaining separating axes
        # are the robot's forward and left directions.
        near = self.nearby(x, y, extent_x, extent_y)
        lo_x, hi_x = self.low_x[near], self.high_x[near]
        lo_y, hi_y = self.low_y[near], self.high_y[near]
        dx, dy = (lo_x + hi_x) / 2 - x, (lo_y + hi_y) / 2 - y
        bx, by = (hi_x - lo_x) / 2, (hi_y - lo_y) / 2
        if np.any((np.abs(dx * c + dy * s) <= half_length + bx * abs(c) + by * abs(s))
                  & (np.abs(-dx * s + dy * c) <= half_width + bx * abs(s) + by * abs(c))):
            return False

        for px, py, radius in pieces:
            offset_x, offset_y = px - x, py - y
            forward = max(0., abs(offset_x * c + offset_y * s) - half_length)
            left = max(0., abs(-offset_x * s + offset_y * c) - half_width)
            if forward * forward + left * left <= radius * radius:
                return False
        return True


def _vec3(values: NDArray[np.float64]) -> Vec3:
    return float(values[0]), float(values[1]), float(values[2])


def layout_visibility(geometry: LayoutGeometry, gamepieces: list[GamepiecePose],
                      keypoints: LayoutKeypoints | None = None) -> LeftCameraVisibility:
    targets: list[SightTarget] = []
    for piece in gamepieces:
        footprints = geometry.cone_footprints if piece.kind == 'cone' else geometry.cube_footprints
        footprint = footprints[int(piece.rotation[1] == 90)]
        pose = transform(piece.position, (0, 0, piece.rotation[2]))
        rotation, translation = pose[:3, :3], pose[:3, 3]
        low, high = np.asarray(footprint.bounds, dtype=np.float64)
        center = (low + high) / 2 @ rotation.T + translation
        half_extent = (high - low) / 2 @ np.abs(rotation).T
        canonical_center = geometry.cone_center if piece.kind == 'cone' else geometry.cube_center
        canonical_points = ([canonical_center] if keypoints is None
                            else [point.position for point in keypoints.for_kind(piece.kind)])
        correction = default_cone().import_rotation if piece.kind == 'cone' else CUBE_IMPORT_ROTATION
        points = world_points(canonical_points, piece.position, piece.rotation, correction)
        targets.append(SightTarget(points, (_vec3(center - half_extent), _vec3(center + half_extent))))
    return LeftCameraVisibility(geometry.occluders, targets)


def _sample_batch(index: int, options: LayoutOptions, geometry: LayoutGeometry,
                  field: _Field, rng: Random, keypoints: LayoutKeypoints | None) -> LayoutBatch:
    for _ in range(MAX_ATTEMPTS):
        count_draw = rng.gauss(options.piece_count_mean, options.piece_count_std)
        if not isfinite(count_draw):
            raise ValueError(f'Batch {index}: sampled a non-finite gamepiece count')
        count = max(0, round(count_draw))
        if count >= options.min_visible:
            break
    else:
        raise ValueError(f'Batch {index}: could not sample a piece count for minimum visibility {options.min_visible}')

    for _ in range(MAX_LAYOUT_ATTEMPTS if options.min_visible else 1):
        batch = _sample_arrangement(index, count, options, geometry, field, rng, keypoints)
        if batch is not None:
            return batch
    raise ValueError(f'Batch {index}: could not find {options.batch_size} poses with '
                     f'at least {options.min_visible} visible gamepieces')


def _sample_arrangement(index: int, count: int, options: LayoutOptions, geometry: LayoutGeometry,
                        field: _Field, rng: Random, keypoints: LayoutKeypoints | None) -> LayoutBatch | None:
    gamepieces: list[GamepiecePose] = []
    circles: list[tuple[float, float, float]] = []
    for item in range(1, count + 1):
        kind: PieceKind = rng.choice(('cone', 'cube'))
        stance = rng.randrange(2)
        footprints = geometry.cone_footprints if kind == 'cone' else geometry.cube_footprints
        footprint = footprints[stance]
        radius = footprint.radius_m
        yaw = rng.random() * 360
        for _ in range(MAX_ATTEMPTS):
            x, y = rng.random() * field.width, rng.random() * field.height
            if field.circle_is_clear(x, y, radius) and all(
                (x - px)**2 + (y - py)**2 > (radius + pr)**2 for px, py, pr in circles
            ):
                circles.append((x, y, radius))
                gamepieces.append(GamepiecePose(kind=kind, position=(x, y, footprint.height_m),
                                                rotation=(0, 90 * stance, yaw)))
                break
        else:
            raise ValueError(f'Batch {index}: could not place gamepiece {item} of {count}')

    visibility = layout_visibility(geometry, gamepieces, keypoints) if options.min_visible else None
    robot_poses: list[Vec3] = []
    for pose_index in range(1, options.batch_size + 1):
        for _ in range(MAX_ATTEMPTS):
            pose = (rng.random() * field.width, rng.random() * field.height, rng.random() * 360)
            if not field.robot_is_clear(pose, options.robot_length / 2, options.robot_width / 2, circles):
                continue
            if visibility is not None and not visibility.meets_minimum(pose, options.min_visible):
                continue
            robot_poses.append(pose)
            break
        else:
            if visibility is not None:
                return None
            raise ValueError(f'Batch {index}: could not place robot pose {pose_index} of {options.batch_size}')
    return LayoutBatch(gamepieces=gamepieces, robot_poses=robot_poses)


def generate_layouts(options: LayoutOptions, geometry: LayoutGeometry, output: Path,
                     keypoints: LayoutKeypoints | None = None) -> Path:
    if keypoints is not None:
        keypoints.for_kind('cone')
        keypoints.for_kind('cube')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output directory already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    field = _Field(geometry)
    rng = Random(options.seed)
    with TemporaryDirectory(prefix='.layouts-', dir=output.parent) as temporary:
        staging = Path(temporary)
        for index in range(1, options.num_batches + 1):
            batch = _sample_batch(index, options, geometry, field, rng, keypoints)
            (staging / f'batch-{index:06d}.json').write_text(batch.model_dump_json(indent=2) + '\n')
        staging.rename(output)
    return output
