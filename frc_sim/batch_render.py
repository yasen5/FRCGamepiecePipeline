"""Render generated layout files with a shared left-camera renderer."""
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .camera import ImageProjector
from .keypoints import KeypointManifest, LayoutKeypoints, PieceKind, SceneLandmarks
from .layouts import CUBE_IMPORT_ROTATION, LayoutBatch
from .model import GamepiecePlacement, Piece, Scene, default_calibration, default_cone
from .renderer import RenderService


def render_layouts(source: Path, output: Path, keypoints: LayoutKeypoints | None = None) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output directory already exists: {output}')
    if not source.is_dir():
        raise ValueError(f'Expected a layout directory: {source}')
    inputs = sorted((int(match[1]), path) for path in source.iterdir()
                    if (match := re.fullmatch(r'batch-(\d+)\.json', path.name)) and path.is_file())
    if not inputs:
        raise ValueError(f'No batch layout JSON files in {source}')
    batches: list[LayoutBatch] = []
    for _, path in inputs:
        try:
            batches.append(LayoutBatch.model_validate_json(path.read_bytes()))
        except ValueError as error:
            raise ValueError(f'Invalid layout {path}: {error}') from error

    classes = ['center']
    if keypoints is not None:
        kinds: set[PieceKind] = {item.kind for batch in batches for item in batch.gamepieces}
        for kind in sorted(kinds):
            keypoints.for_kind(kind)
        classes = sorted({point.name for points in (keypoints.cone, keypoints.cube)
                          for point in points or ()})
    manifest = KeypointManifest(classes=classes, images=[])
    class_ids = {name: index for index, name in enumerate(classes)}
    calibration = default_calibration()
    projector = ImageProjector(calibration, distorted=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    service = RenderService()
    try:
        with TemporaryDirectory(prefix='.render-layouts-', dir=output.parent) as temporary:
            staging = Path(temporary)
            for index, batch in enumerate(batches):
                directory = staging / f'batch_{index}'
                directory.mkdir()
                (directory / 'layout.json').write_text(batch.model_dump_json(indent=2) + '\n')
                pieces: list[Piece] = []
                for item in batch.gamepieces:
                    piece = default_cone() if item.kind == 'cone' else Piece(
                        name='Cube', path='assets/field/model_1.glb', import_rotation=CUBE_IMPORT_ROTATION)
                    piece.position = item.position
                    piece.rotation = item.rotation
                    piece.gamepiece = GamepiecePlacement(theta=item.rotation[2],
                                                         stance='fallen' if item.rotation[1] == 90 else 'upright')
                    if keypoints is not None:
                        piece.keypoints = keypoints.for_kind(item.kind)
                    pieces.append(piece)
                scene = Scene(calibration=calibration, pieces=pieces)
                landmarks = SceneLandmarks.from_scene(scene, named=keypoints is not None)
                for frame, pose in enumerate(batch.robot_poses):
                    scene.robot_xy_heading = pose
                    destination = directory / f'frame_{frame}.png'
                    service.executor.submit(service.export, scene, destination).result()
                    manifest.images.append(landmarks.annotate(
                        scene, projector, destination.relative_to(staging).as_posix(), class_ids))
            manifest.write(staging / 'keypoints.json')
            staging.rename(output)
    finally:
        service.executor.shutdown()
    return output
