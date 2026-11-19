"""Exercise the CLI and dataset writer with only GPU image capture replaced."""
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import pytest

from frc_sim.__main__ import main
from frc_sim.camera import camera_pose
from frc_sim.keypoints import KeypointManifest, LayoutKeypoints
from frc_sim.layouts import GamepiecePose, LayoutBatch
from frc_sim.model import GamepiecePlacement, Keypoint, Piece, Scene, Vec3, scene_json
from frc_sim.renderer import RenderService
from tests.test_camera import calibration


def test_export_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def capture(self: RenderService, state: Scene) -> NDArray[np.uint8]:
        assert state.calibration is not None
        return np.zeros((state.calibration.height, state.calibration.width, 3), dtype=np.uint8)

    def cli(*args: str) -> None:
        monkeypatch.setattr('sys.argv', ['frc-sim', *args])
        main()

    monkeypatch.setattr(RenderService, 'render', capture)
    c = calibration()
    c.extrinsics.rotation_y = 0
    scene = Scene(calibration=c, robot_xy_heading=(0, 0, 0), pieces=[
        Piece(name='Test cone', path='cone.glb', gamepiece=GamepiecePlacement(), position=(4, 0, 0),
              import_rotation=(90, 0, 0), rotation=(0, 90, 0), scale=2, keypoints=[
                  Keypoint(name='corner', position=(.2, .3, .4)),
                  Keypoint(name='corner', position=(.2, .3, -.4)),
                  Keypoint(name='top', position=(0, -5, 0)),
                  Keypoint(name='top', position=(0, 0, 10)),
              ])])
    scene_file = tmp_path / 'scene.json'
    i = c.intrinsics
    optical = np.array([[.9, .7, 4.4], [-.7, .7, 4.4]])
    matrix = np.array([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1.]])
    for distorted in (False, True):
        scene.distortion = distorted
        scene_file.write_text(scene_json(scene, scene_file))
        cli('render', '--scene', str(scene_file), '--output', str(tmp_path / 'single.png'), '--keypoints')
        single = KeypointManifest.model_validate_json((tmp_path / 'single.keypoints.json').read_bytes())
        image, = single.images
        assert image.image == 'single.png'
        with Image.open(tmp_path / image.image) as png:
            assert png.size == (image.width, image.height)
        assert single.classes == ['corner', 'top']
        assert [(point.class_id, point.piece_index) for point in image.points] == [(0, 0), (0, 0)]
        coeffs = np.array([i.k1, i.k2, i.p1, i.p2, i.k3]) if distorted else None
        expected, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), matrix, coeffs)
        np.testing.assert_allclose([(point.x, point.y) for point in image.points], expected.reshape(-1, 2))

    scene.pieces[0].keypoints = []
    scene_file.write_text(scene_json(scene, scene_file))
    missing = tmp_path / 'missing.png'
    with pytest.raises(SystemExit, match='2'):
        cli('render', '--scene', str(scene_file), '--output', str(missing), '--keypoints')
    assert not missing.exists()

    layouts = tmp_path / 'layouts'
    layouts.mkdir()
    pose: Vec3 = (3, 3, 0)
    camera = camera_pose(Scene(robot_xy_heading=pose))
    pieces = []
    for depth in (2, 4):
        x, y, z = camera[:3, 3] + depth * camera[:3, 0]
        pieces.append(GamepiecePose(kind='cube', position=(float(x), float(y), float(z)), rotation=(0, 0, 0)))
    batch = LayoutBatch(gamepieces=pieces, robot_poses=[pose, (3, 3, 180)])
    (layouts / 'batch-000001.json').write_text(batch.model_dump_json())
    definitions = tmp_path / 'landmarks.json'
    output = tmp_path / 'renders'
    args = ('render-layouts', '--layouts', str(layouts), '--output', str(output), '--keypoints')
    with pytest.raises(SystemExit, match='2'):
        cli(*args)
    assert not output.exists()
    definitions.write_text(LayoutKeypoints().model_dump_json())
    with pytest.raises(SystemExit, match='2'):
        cli(*args, '--keypoints-file', str(definitions))
    assert not output.exists()
    definitions.write_text(LayoutKeypoints(cube=[Keypoint(name='corner', position=(0, 0, 0))]).model_dump_json())
    cli(*args, '--keypoints-file', str(definitions))
    manifest = KeypointManifest.model_validate_json((output / 'keypoints.json').read_bytes())
    assert manifest.classes == ['corner']
    first, second = manifest.images
    assert [frame.image for frame in manifest.images] == ['batch_0/frame_0.png', 'batch_0/frame_1.png']
    assert all((output / frame.image).is_file() for frame in manifest.images)
    assert [point.piece_index for point in first.points] == [0, 1]
    np.testing.assert_allclose([(point.x, point.y) for point in first.points], [(637.528862, 419.464139)] * 2)
    assert second.points == []
