"""Render generated layout files with a shared left-camera renderer."""
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .layouts import CUBE_IMPORT_ROTATION, LayoutBatch
from .model import GamepiecePlacement, Piece, Scene, default_calibration, default_cone
from .renderer import RenderService


def render_layouts(source: Path, output: Path) -> Path:
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
                    pieces.append(piece)
                scene = Scene(calibration=default_calibration(), pieces=pieces)
                for frame, pose in enumerate(batch.robot_poses):
                    scene.robot_xy_heading = pose
                    service.executor.submit(service.export, scene, directory / f'frame_{frame}.png').result()
            staging.rename(output)
    finally:
        service.executor.shutdown()
    return output
