"""Field editing geometry (meters and degrees)."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import trimesh
from .camera import transform
from .model import Piece


@lru_cache(maxsize=32)
def vertices(path: str, modified: int) -> NDArray[np.float64]:
    """Flatten GLB node transforms before calculating contact; modified invalidates cache."""
    mesh = trimesh.load_scene(path).to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('Gamepiece must contain triangle mesh geometry')
    return np.asarray(mesh.vertices, dtype=np.float64)


def place(piece: Piece, x: float, y: float) -> None:
    placement = piece.gamepiece
    if placement is None:
        raise ValueError('Tag this asset as a gamepiece before placing it')
    roll, pitch = placement.upright_tilt if placement.stance == 'upright' else placement.fallen_tilt
    rotation = (roll, pitch, placement.theta)
    matrix = transform((0, 0, 0), rotation) @ transform((0, 0, 0), piece.import_rotation)
    path = Path(piece.path)
    points = vertices(str(path), path.stat().st_mtime_ns) * piece.scale
    height = -float((points @ matrix[:3, :3].T)[:, 2].min())
    piece.rotation = rotation
    piece.position = (x, y, height)
