"""Validated domain records. Coordinates: meters, NWU; Euler angles in degrees."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

Vec3 = tuple[float, float, float]


class Record(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", validate_assignment=True)


class Extrinsics(Record):
    translation_x: float
    translation_y: float
    translation_z: float
    rotation_x: float
    rotation_y: float
    rotation_z: float


class Intrinsics(Record):
    fx: PositiveFloat
    fy: PositiveFloat
    cx: float
    cy: float
    k1: float = 0
    k2: float = 0
    k3: float = 0
    p1: float = 0
    p2: float = 0


class Calibration(Record):
    # Acquisition metadata from frc971/cos is intentionally ignored at this boundary.
    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")
    width: Annotated[int, Field(gt=0, le=8192)]
    height: Annotated[int, Field(gt=0, le=8192)]
    extrinsics: Extrinsics
    intrinsics: Intrinsics


class Piece(Record):
    name: str
    path: str
    position: Vec3 = (2, 2, 0)
    rotation: Vec3 = (0, 0, 0)
    import_rotation: Vec3 = (0, 0, 0)
    scale: PositiveFloat = 1
    color: Vec3 = (0.8, 0.65, 0.1)


DEFAULT_CONE_PATH = "2023ChargedUpConeChopped.glb"


def default_cone() -> Piece:
    """User cone: meters, tip along -Z, base underside at source Z=+0.0127."""
    return Piece(name="Chopped cone", path=DEFAULT_CONE_PATH,
                 position=(4, 4, 0.0127), import_rotation=(180, 0, 0))


class Light(Record):
    kind: Literal["directional", "point", "spot"] = "directional"
    position: Vec3 = (4, 4, 6)
    direction: Vec3 = (0.3, 0.4, -1)
    color: Vec3 = (1, 1, 1)
    intensity: Annotated[float, Field(ge=0)] = 1
    shadow: bool = True
    inner_degrees: Annotated[float, Field(gt=0, lt=179)] = 30
    outer_degrees: Annotated[float, Field(gt=0, lt=179)] = 60


class Scene(Record):
    version: Literal[1] = 1
    field_dir: str = "assets/field"
    calibration: Calibration | None = None
    robot_xy_heading: Vec3 = (2, 4, 0)
    pieces: list[Piece] = Field(default_factory=lambda: [default_cone()])
    lights: list[Light] = Field(default_factory=lambda: [Light()])
    ambient: Vec3 = (0.2, 0.2, 0.2)
    distortion: bool = True
    inspect_target: Vec3 = (8.27, 4, 0)
    inspect_yaw: float = 225
    inspect_pitch: Annotated[float, Field(ge=-89, le=89)] = 35
    inspect_distance: PositiveFloat = 15


def load_scene(path: Path) -> Scene:
    """Resolve asset paths relative to the scene document, never the shell cwd."""
    scene = Scene.model_validate_json(path.read_text())
    scene.field_dir = str((path.parent / scene.field_dir).resolve())
    for piece in scene.pieces:
        piece.path = str((path.parent / piece.path).resolve())
    return scene


def scene_json(scene: Scene, destination: Path) -> str:
    import os
    result = scene.model_copy(deep=True)
    result.field_dir = os.path.relpath(scene.field_dir, destination.parent)
    for piece in result.pieces:
        piece.path = os.path.relpath(piece.path, destination.parent)
    return result.model_dump_json(indent=2)
