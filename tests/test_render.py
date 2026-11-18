"""Explicit opt-in full GPU pipeline; failures must not silently skip as successes."""
import os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from frc_sim.model import Piece, Scene, default_cone, load_scene
from frc_sim.renderer import RenderService
from tests.test_camera import calibration


@pytest.mark.skipif(os.environ.get('FRC_GPU_TEST') != '1', reason='Set FRC_GPU_TEST=1 on a Vulkan GPU host')
def test_full_pipeline(tmp_path: Path) -> None:
    state = Scene(calibration=calibration(), pieces=[
        default_cone(),
        Piece(name='cube', path='assets/field/model_1.glb', position=(4, 4.5, 0), import_rotation=(90, 0, 0))])
    service = RenderService()
    try:
        path = tmp_path/'frame.png'
        service.executor.submit(service.export, state, path).result()
        first = np.asarray(Image.open(path))
        assert first.shape == (800, 1280, 3)
        assert first.std() > 1
        reloaded = load_scene(path.with_suffix('.json'))
        second = service.executor.submit(service.render, reloaded).result()
        np.testing.assert_allclose(first, second, atol=2)
        state.robot_xy_heading = (2.5, 4, 10)
        state.pieces[0].rotation = (0, 0, 45)
        state.lights[0].direction = (-.5, .4, -1)
        changed = service.executor.submit(service.render, state).result()
        assert np.mean(np.abs(first.astype(float)-changed.astype(float))) > .1
    finally:
        service.executor.shutdown()
