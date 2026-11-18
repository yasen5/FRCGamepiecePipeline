"""Placement-to-document pipeline and pivotal driving integration."""
from pathlib import Path
import math
import numpy as np
import pytest
from frc_sim.camera import transform
from frc_sim.interaction import drive, place, vertices
from frc_sim.model import Scene, Piece, GamepiecePlacement, default_cone, load_scene, scene_json


@pytest.mark.parametrize('asset', ['cone', 'cube'])
def test_placement_document_pipeline(tmp_path: Path, asset: str) -> None:
    piece = default_cone() if asset == 'cone' else Piece(name='Cube',
        path='assets/field/model_1.glb', import_rotation=(90, 0, 0), gamepiece=GamepiecePlacement())
    assert piece.gamepiece is not None
    for scale in (.5, 1.5):
        piece.scale = scale
        for stance in ('upright', 'fallen'):
            piece.gamepiece.stance = stance
            for theta in (0., 63., 180.):
                piece.gamepiece.theta = theta
                place(piece, 4, 3)
                matrix = transform(piece.position, piece.rotation) @ transform((0, 0, 0), piece.import_rotation)
                points = vertices(piece.path, Path(piece.path).stat().st_mtime_ns)*scale
                world = points @ matrix[:3, :3].T + matrix[:3, 3]
                assert world[:, 2].min() == pytest.approx(0, abs=1e-10)
                destination = tmp_path/'scene.json'
                destination.write_text(scene_json(Scene(pieces=[piece]), destination))
                restored = load_scene(destination).pieces[0]
                assert restored.gamepiece == piece.gamepiece
                assert restored.position == piece.position
                assert restored.rotation == piece.rotation


def test_robot_relative_driving() -> None:
    assert drive((0, 0, 90), 1, 0, 0, 1, 1, 60) == pytest.approx((0, 1, 90))
    assert drive((0, 0, 90), 0, 1, 0, 1, 1, 60) == pytest.approx((-1, 0, 90))
    x, y, heading = drive((0, 0, 0), 1, 1, 1, .5, 2, 60)
    assert math.hypot(x, y) == pytest.approx(1)
    assert heading == 30
    assert drive((1, 2, 30), 0, 0, -1, .5, 1, 60) == (1, 2, 0)
    pose = (0., 0., 0.)
    for _ in range(10):
        pose = drive(pose, 1, 0, 0, .1, 1, 60)
    assert pose == pytest.approx((1, 0, 0))
