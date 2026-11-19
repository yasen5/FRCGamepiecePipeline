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


class GamepiecePlacement(Record):
    """Tilts in the import-corrected frame; theta is CCW field yaw in degrees."""
    upright_tilt: tuple[float, float] = (0, 0)
    fallen_tilt: tuple[float, float] = (0, 90)
    stance: Literal['upright', 'fallen'] = 'upright'
    theta: float = 0


class Piece(Record):
    gamepiece: GamepiecePlacement | None = None
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
                 position=(4, 4, 0.0127), import_rotation=(180, 0, 0), gamepiece=GamepiecePlacement())


class Light(Record):
    kind: Literal["directional", "point", "spot", "area"] = "directional"
    position: Vec3 = (4, 4, 6)
    direction: Vec3 = (0.3, 0.4, -1)
    color: Vec3 = (1, 1, 1)
    intensity: Annotated[float, Field(ge=0)] = 1
    width: PositiveFloat = 1.5
    height: PositiveFloat = 0.8
    shadow: bool = True
    inner_degrees: Annotated[float, Field(gt=0, lt=179)] = 30
    outer_degrees: Annotated[float, Field(gt=0, lt=179)] = 60


def default_calibration() -> Calibration:
    # frc971/cos constants/second_bot/left_camera.json, retrieved 2026-11-18.
    return Calibration(width=1280, height=800,
        extrinsics=Extrinsics(translation_x=.2897, translation_y=.3608, translation_z=.3434,
                             rotation_x=0, rotation_y=-15, rotation_z=45),
        intrinsics=Intrinsics(fx=901.1890743775721, fy=901.0617429863127,
            cx=637.5288620693618, cy=419.4641386227522, k1=.03863073515655131,
            k2=-.07808454691737804, k3=.02727197961030114,
            p1=.0033423455589040776, p2=.00036377102345917046))


class RenderSettings(Record):
    samples_per_pixel: Annotated[int, Field(ge=64, le=1024)] = 256
    path_depth: Annotated[int, Field(ge=1, le=32)] = 8


def indoor_lights() -> list[Light]:
    return [Light(kind='area', position=(x, y, 8), direction=(0, 0, -1), intensity=20)
            for x in (3., 8.27, 13.54) for y in (2., 6.)]


class Scene(Record):
    version: Literal[1] = 1
    field_dir: str = "assets/field"
    calibration: Calibration | None = Field(default_factory=default_calibration)
    robot_xy_heading: Vec3 = (2, 4, 0)
    pieces: list[Piece] = Field(default_factory=lambda: [default_cone()])
    lights: list[Light] = Field(default_factory=indoor_lights)
    ambient: Vec3 = (0.03, 0.03, 0.03)
    render_settings: RenderSettings = Field(default_factory=RenderSettings)
    distortion: bool = True
    inspect_target: Vec3 = (8.27, 4, 0)
    inspect_yaw: float = 225
    inspect_pitch: Annotated[float, Field(ge=-89, le=89)] = 35
    inspect_distance: PositiveFloat = 15


def load_scene(path: Path) -> Scene:
    """Resolve asset paths relative to the scene document, never the shell cwd."""
    scene = Scene.model_validate_json(path.read_text())
    if scene.calibration is None:
        scene.calibration = default_calibration()
    scene.field_dir = str((path.parent / scene.field_dir).resolve())
    for piece in scene.pieces:
        piece.path = str((path.parent / piece.path).resolve())
    return scene


def scene_json(scene: Scene, destination: Path) -> str:
    import os
    result = scene.model_copy(deep=True)
    parent = destination.parent.resolve()
    result.field_dir = os.path.relpath(Path(scene.field_dir).resolve(), parent)
    for piece in result.pieces:
        piece.path = os.path.relpath(Path(piece.path).resolve(), parent)
    return result.model_dump_json(indent=2)
