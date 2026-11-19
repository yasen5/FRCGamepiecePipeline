"""CPU coverage of cached geometry, batch serialization, and placement."""
from itertools import combinations
from math import cos, hypot, radians, sin
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from frc_sim.camera import camera_pose
from frc_sim.layouts import LayoutBatch, LayoutOptions, generate_layouts, layout_visibility, load_geometry
from frc_sim.model import Scene


def test_layout_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    field_dir = tmp_path / 'assets'
    field_dir.mkdir()
    (field_dir / 'config.json').write_text("""{
        "widthInches": 300, "heightInches": 300, "rotations": [],
        "gamePieces": [{"stagedObjects": ["Staged_Cube"]}]
    }""")
    field = trimesh.Scene()
    for name, size, center in [
        ('carpet', (8, 8, .01), (0, 0, -.005)),
        ('obstacle', (2, 2, .5), (0, 0, .25)),
        ('Staged Cube', (.2, .2, .2), (2, 2, .1)),
    ]:
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        field.add_geometry(mesh, node_name=name, geom_name=name)
    field.export(str(field_dir / 'model.glb'))
    cone_path = field_dir / 'cone.glb'
    cone = trimesh.creation.box(extents=(.2, .2, .4))
    cone.apply_translation((0, 0, -.15))
    cone.export(str(cone_path))
    trimesh.creation.box(extents=(.25, .25, .25)).export(str(field_dir / 'model_1.glb'))

    cache = tmp_path / 'cache'
    geometry = load_geometry(cache, field_dir, cone_path)
    assert np.asarray(geometry.obstacles) == pytest.approx(np.array([[2.81, 2.81, 4.81, 4.81]]))
    assert [footprint.height_m for footprint in geometry.cone_footprints] == pytest.approx([.05, .1])

    def unexpected_mesh_load(*args: object, **kwargs: object) -> None:
        pytest.fail('Warm geometry loading opened a mesh')

    monkeypatch.setattr(trimesh, 'load_scene', unexpected_mesh_load)
    warm = load_geometry(cache, field_dir, cone_path)
    assert warm == geometry
    options = LayoutOptions(batch_size=40, num_batches=4, robot_length=1., robot_width=.6, seed=73)
    first = generate_layouts(options, geometry, tmp_path / 'first')
    second = generate_layouts(options, warm, tmp_path / 'second')
    batches = sorted(first.glob('*.json'))
    assert len(batches) == 4
    counts: set[int] = set()
    variants: set[tuple[str, float]] = set()
    obstacle = np.array([[2.81, 2.81], [4.81, 2.81], [4.81, 4.81], [2.81, 4.81]], dtype=np.float32)
    for document in batches:
        assert document.read_bytes() == (second / document.name).read_bytes()
        batch = LayoutBatch.model_validate_json(document.read_bytes())
        assert len(set(batch.robot_poses)) == 40
        counts.add(len(batch.gamepieces))
        circles: list[tuple[float, float, float]] = []
        for piece in batch.gamepieces:
            x, y, z = piece.position
            footprints = geometry.cone_footprints if piece.kind == 'cone' else geometry.cube_footprints
            pitch = piece.rotation[1]
            assert pitch in (0, 90)
            variants.add((piece.kind, pitch))
            footprint = footprints[int(pitch / 90)]
            assert z == footprint.height_m
            radius = footprint.radius_m
            assert radius < x < 7.62 - radius and radius < y < 7.62 - radius
            assert cv2.pointPolygonTest(obstacle, (x, y), True) < -radius
            circles.append((x, y, radius))
        for (x, y, radius), (px, py, other_radius) in combinations(circles, 2):
            assert hypot(x - px, y - py) > radius + other_radius

        for x, y, theta in batch.robot_poses:
            angle = radians(theta)
            corners = np.array([(x + dx * cos(angle) - dy * sin(angle),
                                 y + dx * sin(angle) + dy * cos(angle))
                                for dx, dy in [(-.5, -.3), (.5, -.3), (.5, .3), (-.5, .3)]],
                               dtype=np.float32)
            assert np.all((corners > 0) & (corners < 7.62))
            assert cv2.intersectConvexConvex(corners, obstacle)[0] == 0
            for px, py, radius in circles:
                assert cv2.pointPolygonTest(corners, (px, py), True) < -radius
        pose = camera_pose(Scene(robot_xy_heading=batch.robot_poses[0]))
        assert pose[2, 3] == pytest.approx(.3434)
    assert len(counts) > 1
    assert variants == {('cone', 0), ('cone', 90), ('cube', 0), ('cube', 90)}

    visible_options = LayoutOptions(5, num_batches=2, seed=73, min_visible=2)
    visible = generate_layouts(visible_options, warm, tmp_path / 'visible')
    repeated = generate_layouts(visible_options, warm, tmp_path / 'visible-repeated')
    for document in sorted(visible.glob('*.json')):
        assert document.read_bytes() == (repeated / document.name).read_bytes()
        batch = LayoutBatch.model_validate_json(document.read_bytes())
        visibility = layout_visibility(warm, batch.gamepieces)
        assert all(visibility.meets_minimum(pose, 2) for pose in batch.robot_poses)

    empty_dir = generate_layouts(LayoutOptions(1, piece_count_mean=0, piece_count_std=0, seed=73),
                                 geometry, tmp_path / 'empty')
    assert LayoutBatch.model_validate_json(next(empty_dir.glob('*.json')).read_bytes()).gamepieces == []
    impossible = tmp_path / 'impossible'
    with pytest.raises(ValueError, match='Batch 1: could not place robot pose 1'):
        generate_layouts(LayoutOptions(1, robot_length=100), geometry, impossible)
    assert not impossible.exists()


def test_left_camera_visibility() -> None:
    from frc_sim.camera import robot_camera_pose
    from frc_sim.model import Vec3, default_calibration
    from frc_sim.visibility import LeftCameraVisibility, SightTarget

    pose: Vec3 = (2, 3, 20)
    camera = robot_camera_pose(pose, default_calibration())

    def target(forward: float, left: float, up: float) -> SightTarget:
        point = camera[:3, :3] @ np.array([forward, left, up]) + camera[:3, 3]
        x, y, z = (float(v) for v in point)
        return SightTarget(point.reshape(1, 3), ((x - .05, y - .05, z - .05),
                                                  (x + .05, y + .05, z + .05)))

    near, far = target(2, 0, 0), target(4, 0, 0)
    outside = [target(-2, 0, 0), target(4, 3, 0), target(4, 0, 2)]
    assert LeftCameraVisibility([], [far, *outside]).meets_minimum(pose, 1)
    assert not LeftCameraVisibility([], outside).meets_minimum(pose, 1)
    assert not LeftCameraVisibility([near.bounds], [far]).meets_minimum(pose, 1)
    assert not LeftCameraVisibility([], [near, far]).meets_minimum(pose, 2)
    assert LeftCameraVisibility([], [near, target(4, 1, 0)]).meets_minimum(pose, 2)
    with pytest.raises(ValueError, match='fixed gamepiece count'):
        LayoutOptions(5, piece_count_mean=1, piece_count_std=0, min_visible=2)
