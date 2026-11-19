"""SAPIEN boundary. Its incomplete extension stubs require localized Any handles."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .camera import camera_pose, inspection_pose, projection, transform
from .field import FieldConfig
from .model import DEFAULT_CONE_PATH, GamepiecePlacement, Scene, Vec3, scene_json


class Renderer:
    def __init__(self) -> None:
        # SAPIEN/pybind handles stay within this adapter, never serialized or used as domain data.
        import sapien
        self.api: Any = sapien
        # 3.0.3 renders offscreen without constructing a Viewer or window.
        self.scene: Any = None
        self.signature = ''
        self.piece_poses: tuple[tuple[Vec3, Vec3], ...] = ()
        self.actors: list[Any] = []
        self.camera: Any = None
        self.camera_size = (0, 0)

    def rebuild(self, state: Scene) -> None:
        root = Path(state.field_dir)
        stock_gamepieces = {Path(DEFAULT_CONE_PATH).resolve(), (root / 'model_0.glb').resolve(),
                            (root / 'model_1.glb').resolve()}
        matte_pieces = [piece.gamepiece is not None or Path(piece.path).resolve() in stock_gamepieces
                        for piece in state.pieces]
        static = state.model_copy(deep=True)
        for piece, matte in zip(static.pieces, matte_pieces):
            # Only the gamepiece tag affects the material, not the placement settings.
            piece.gamepiece = GamepiecePlacement() if matte else None
            piece.position = (0, 0, 0)
            piece.rotation = (0, 0, 0)
        signature = static.model_dump_json(exclude={'robot_xy_heading', 'calibration', 'distortion',
                                                  'inspect_target', 'inspect_yaw', 'inspect_pitch', 'inspect_distance', 'render_settings'})
        if signature == self.signature:
            return
        self.scene = self.api.Scene()
        self.camera = None
        self.camera_size = (0, 0)
        self.actors = []
        config = FieldConfig.model_validate_json((root / 'config.json').read_text())
        field_pose = config.world_transform()
        # Remove staged objects by GLB node name while preserving scene hierarchy/materials.
        import hashlib
        cache = Path('runtime') / ('field-' + hashlib.sha256(
            (root / 'model.glb').read_bytes() + (root / 'config.json').read_bytes()
            + b'matte-carpet-and-gamepieces-v2').hexdigest()[:16] + '.glb')
        if not cache.exists():
            import trimesh
            from trimesh.visual.material import PBRMaterial
            from trimesh.visual.texture import TextureVisuals

            model = trimesh.load_scene(str(root / 'model.glb'))
            for node in config.staged_nodes(list(model.graph.nodes_geometry)):
                model.graph.transforms.remove_node(node)
            matte_meshes = {'Carpet (FE-2023-01)', 'Traffic Cone v2', 'Cube v3'}
            for name, mesh in model.geometry.items():
                # Trimesh appends numeric suffixes to duplicate mesh names.
                base, _, suffix = name.rpartition('_')
                if name not in matte_meshes and not (suffix.isdecimal() and base in matte_meshes):
                    continue
                visual = mesh.visual
                if isinstance(visual, TextureVisuals) and isinstance(visual.material, PBRMaterial):
                    # Other field meshes can share the original material.
                    material = visual.material.copy()
                    material.metallicFactor = 0.0
                    material.roughnessFactor = 1.0
                    material.metallicRoughnessTexture = None
                    visual.material = material
            cache.parent.mkdir(parents=True, exist_ok=True)
            model.export(str(cache))
        self.add_mesh(cache, field_pose, 1, None)
        for piece, matte in zip(state.pieces, matte_pieces):
            path = Path(piece.path)
            self.add_mesh(path, transform(piece.position, piece.rotation) @
                          transform((0, 0, 0), piece.import_rotation), piece.scale,
                          piece.color if path.suffix.lower() == '.stl' else None, matte=matte)
        self.scene.set_ambient_light(state.ambient)
        for light in state.lights:
            if light.kind != 'point' and np.linalg.norm(light.direction) < 1e-8:
                raise ValueError('Light direction must be nonzero')
            color = [v*light.intensity for v in light.color]
            common = dict(shadow=light.shadow, shadow_near=.02, shadow_far=60)
            if light.kind == 'area':
                # SAPIEN's light forward axis is +X, as for spot lights.
                rotation = self.api.math.shortest_rotation([1, 0, 0], light.direction)
                self.scene.add_area_light_for_ray_tracing(
                    self.api.Pose(light.position, rotation), color, light.width/2, light.height/2)
            elif light.kind == 'directional':
                self.scene.add_directional_light(light.direction, color, position=light.position, shadow_scale=25, **common)
            elif light.kind == 'point':
                self.scene.add_point_light(light.position, color, **common)
            else:
                if light.inner_degrees > light.outer_degrees:
                    raise ValueError('Spotlight inner angle must not exceed outer angle')
                self.scene.add_spot_light(light.position, light.direction,
                                         np.deg2rad(light.inner_degrees), np.deg2rad(light.outer_degrees), color, **common)
        self.piece_poses = tuple((piece.position, piece.rotation) for piece in state.pieces)
        self.signature = signature

    def add_mesh(self, path: Path, pose: NDArray[np.float64], scale: float,
                 color: tuple[float, float, float] | None, *, matte: bool = False) -> None:
        if path.suffix.lower() not in {'.glb', '.stl'} or not path.is_file():
            raise ValueError(f'Expected an existing GLB/STL: {path}')
        builder = self.scene.create_actor_builder()
        material = None
        if color is not None:
            material = self.api.render.RenderMaterial()
            material.base_color = [*color, 1]
        builder.add_visual_from_file(str(path), scale=[scale]*3, material=material)
        actor = builder.build_kinematic()
        if matte:
            from sapien.render import RenderBodyComponent

            for component in actor.components:
                if isinstance(component, RenderBodyComponent):
                    for shape in component.render_shapes:
                        for part in shape.parts:
                            part_material = part.material
                            # SAPIEN accepts None to clear textures despite the stub annotations.
                            part_material.metallic_texture = None  # pyright: ignore[reportAttributeAccessIssue]
                            part_material.roughness_texture = None  # pyright: ignore[reportAttributeAccessIssue]
                            part_material.metallic = 0.0
                            part_material.roughness = 1.0
                            part_material.specular = 0.0
        actor.set_pose(self.api.Pose(pose))
        self.actors.append(actor)

    def add_camera(self, name: str, width: int, height: int, shader: str) -> Any:
        # pybind camera/entity handles are confined to this SAPIEN adapter.
        # Unlike set_camera_shader_dir, this constructor does not resolve pack names.
        shader_path = files('sapien').joinpath('vulkan_shader', shader)
        entrypoint = 'camera.rgen' if shader == 'rt' else 'gbuffer.vert'
        if not shader_path.joinpath(entrypoint).is_file():
            raise RuntimeError(f'Missing SAPIEN shader entrypoint: {shader_path}/{entrypoint}')
        camera = self.api.render.RenderCameraComponent(width, height, str(shader_path))
        entity = self.api.Entity()
        entity.name = name
        entity.add_component(camera)
        self.scene.add_entity(entity)
        camera.near, camera.far = .02, 100
        return camera

    def render(self, state: Scene, preview: bool = False, inspection: bool = False) -> NDArray[np.uint8]:
        self.api.render.set_ray_tracing_samples_per_pixel(state.render_settings.samples_per_pixel)
        self.api.render.set_ray_tracing_path_depth(state.render_settings.path_depth)
        self.api.render.set_ray_tracing_denoiser('none')
        self.rebuild(state)
        poses = tuple((piece.position, piece.rotation) for piece in state.pieces)
        if poses != self.piece_poses:
            for actor, piece, pose, previous in zip(self.actors[1:], state.pieces, poses, self.piece_poses):
                if pose != previous:
                    actor.set_pose(self.api.Pose(transform(*pose) @ transform((0, 0, 0), piece.import_rotation)))
            self.piece_poses = poses
        if inspection:
            from .camera import Projection
            p = Projection(960, 540, 700, 700, 480, 270)
            pose = inspection_pose(state)
        else:
            if state.calibration is None:
                raise ValueError('Load a combined camera JSON to render the robot camera')
            p = projection(state.calibration.model_dump_json(), preview, state.distortion)
            pose = camera_pose(state)
        if self.camera_size != (p.width, p.height):
            if self.camera is not None:
                self.scene.remove_camera(self.camera)
            self.camera = self.add_camera('output', p.width, p.height, 'rt')
            self.camera_size = (p.width, p.height)
        self.camera.set_perspective_parameters(.02, 100, p.fx, p.fy, p.cx, p.cy, 0)
        self.camera.entity.set_pose(self.api.Pose(pose))
        self.scene.update_render()
        self.camera.take_picture()
        frame = np.asarray(self.camera.get_picture('Color'))[..., :3]
        rgb = (np.clip(frame, 0, 1)*255).astype(np.uint8)
        if p.map_x is not None and p.map_y is not None:
            rgb = cv2.remap(rgb, p.map_x, p.map_y, cv2.INTER_LINEAR)
        return np.asarray(rgb, dtype=np.uint8)


class RenderService:
    """One worker owns all GPU state; each request produces a single still."""
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sapien')
        self.renderer: Renderer | None = None

    def render(self, state: Scene, preview: bool = False, inspection: bool = False) -> NDArray[np.uint8]:
        if self.renderer is None:
            self.renderer = Renderer()
        return self.renderer.render(state, preview, inspection)

    def field_map(self, state: Scene, width: int, height: int) -> NDArray[np.uint8]:
        if self.renderer is None:
            self.renderer = Renderer()
        renderer = self.renderer
        background = state.model_copy(deep=True)
        background.pieces = []
        from .model import Light
        background.lights = [Light()]
        background.ambient = (.3, .3, .3)
        renderer.rebuild(background)
        config = FieldConfig.model_validate_json((Path(state.field_dir)/'config.json').read_text())
        w, h = config.size_m
        camera = renderer.add_camera('field-map', width, height, 'default')
        try:
            camera.set_orthographic_parameters(.02, 100, -w/2, w/2, -h/2, h/2)
            # Camera forward is -world Z, camera left is -world X, camera up is +world Y.
            pose = np.eye(4)
            pose[:3, :3] = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
            pose[:3, 3] = (w/2, h/2, 20)
            camera.entity.set_pose(renderer.api.Pose(pose))
            renderer.scene.update_render()
            camera.take_picture()
            rgb = np.asarray(camera.get_picture('Color'))[..., :3]
            return (np.clip(rgb, 0, 1)*255).astype(np.uint8)
        finally:
            renderer.scene.remove_camera(camera)

    def export(self, state: Scene, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.render(state)).save(destination)
        destination.with_suffix('.json').write_text(scene_json(state, destination.with_suffix('.json')))
        return destination
