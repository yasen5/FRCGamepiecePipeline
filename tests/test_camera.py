"""Pivotal calibration tests independent of a functioning GPU."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from frc_sim.camera import camera_pose, projection
from frc_sim.model import Calibration, Extrinsics, Intrinsics, Scene, load_scene, scene_json, Piece


def calibration() -> Calibration:
    return Calibration(width=1280, height=800,
        extrinsics=Extrinsics(translation_x=.2, translation_y=.1, translation_z=.3,
                              rotation_x=0, rotation_y=-20, rotation_z=0),
        intrinsics=Intrinsics(fx=900, fy=910, cx=650, cy=403, k1=.04, k2=-.1, k3=.05, p1=.001, p2=-.002))


def test_pose_and_projection_pipeline() -> None:
    c = calibration()
    state = Scene(calibration=c, robot_xy_heading=(2, 3, 90))
    pose = camera_pose(state)
    np.testing.assert_allclose(pose[:3, 3], [1.9, 3.2, .3])
    assert pose[2, 0] > 0  # negative mount pitch looks upward in NWU
    p = projection(c.model_dump_json(), False, True)
    assert p.map_x is not None and p.map_y is not None
    # Recover ideal source coordinates from the overscan map, then distort forward.
    yy, xx = np.mgrid[0:800:80, 0:1280:128]
    coords = np.stack([p.map_x[yy, xx], p.map_y[yy, xx]], axis=-1)
    rays = np.concatenate([(coords-[p.cx, p.cy])/[p.fx, p.fy], np.ones((*coords.shape[:2], 1))], axis=-1)
    i = c.intrinsics
    k = np.array([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1.]])
    output, _ = cv2.projectPoints(rays.reshape(-1, 3), np.zeros(3), np.zeros(3), k,
                                 np.array([i.k1, i.k2, i.p1, i.p2, i.k3]))
    np.testing.assert_allclose(output.reshape(coords.shape), np.stack([xx, yy], axis=-1), atol=.01)
    preview = projection(c.model_dump_json(), True, False)
    assert preview.height == 540
    assert preview.fx / i.fx == pytest.approx(preview.width / c.width)
    assert preview.cy / i.cy == pytest.approx(preview.height / c.height)


def test_scene_roundtrip_and_invalid_calibration(tmp_path: Path) -> None:
    state = Scene(calibration=calibration(), pieces=[Piece(name='cone', path='assets/field/model_0.glb')])
    destination = tmp_path / 'scene.json'
    destination.write_text(scene_json(state, destination))
    loaded = load_scene(destination)
    assert loaded.calibration == state.calibration
    assert loaded.pieces[0].position == state.pieces[0].position
    assert Path(loaded.pieces[0].path) == Path(state.pieces[0].path).resolve()
    with pytest.raises(ValidationError):
        Calibration.model_validate_json('{"width":0}')
