"""Single shared scene browser UI. Access through a localhost SSH tunnel."""
from __future__ import annotations

import asyncio
import base64
import io
import time
from pathlib import Path
from uuid import uuid4

from nicegui import events, ui
from PIL import Image

from .model import Calibration, Light, Piece, Scene, default_cone, load_scene, scene_json
from .renderer import RenderService


def launch(state: Scene, host: str, port: int) -> None:
    service = RenderService()
    def page() -> None:
        nonlocal state
        busy = False
        revision = 0
        rendered_revision = -1
        inspection = True
        selected = 0
        last_mouse: tuple[float, float] | None = None

        def dirty() -> None:
            nonlocal revision
            revision += 1

        def set_vector(record: object, field: str, index: int, value: float | None) -> None:
            if value is None:
                return
            vector = list(getattr(record, field))
            vector[index] = float(value)
            setattr(record, field, tuple(vector))
            dirty()

        def vector_controls(record: object, field: str, labels: tuple[str, str, str]) -> None:
            with ui.row():
                for index, label in enumerate(labels):
                    ui.number(label, value=getattr(record, field)[index],
                              on_change=lambda e, j=index: set_vector(record, field, j, e.value)).classes('w-28')

        async def upload(e: events.UploadEventArguments) -> None:
            nonlocal selected, state
            try:
                name = Path(e.file.name).name
                data = await e.file.read()
                suffix = Path(name).suffix.lower()
                if suffix == '.json':
                    calibration = Calibration.model_validate_json(data)
                    state.calibration = calibration
                    ui.notify(f'Loaded calibration: {calibration.width} × {calibration.height}')
                elif suffix in {'.glb', '.stl'}:
                    path = Path('runtime') / f'{uuid4().hex}-{name}'
                    path.write_bytes(data)
                    state.pieces.append(Piece(name=name, path=str(path)))
                    selected = len(state.pieces)-1
                    objects.refresh()
                else:
                    raise ValueError('Upload a combined calibration JSON, GLB, or STL')
                dirty()
            except Exception as error:
                ui.notify(str(error), type='negative')

        def load_document() -> None:
            nonlocal state
            try:
                state = load_scene(Path(document.value))
                objects.refresh()
                lighting.refresh()
                robot.refresh()
                inspection_controls.refresh()
                dirty()
            except Exception as error:
                ui.notify(str(error), type='negative')

        def save_document() -> None:
            try:
                path = Path(document.value)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(scene_json(state, path))
                ui.download.file(path)
            except Exception as error:
                ui.notify(str(error), type='negative')

        async def export() -> None:
            try:
                snapshot = state.model_copy(deep=True)
                path = Path('runtime') / f'export-{uuid4().hex}.png'
                await asyncio.get_running_loop().run_in_executor(service.executor, service.export, snapshot, path)
                ui.download.file(path)
                ui.download.file(path.with_suffix('.json'))
            except Exception as error:
                ui.notify(str(error), type='negative')

        def change_view(e: events.ValueChangeEventArguments) -> None:
            nonlocal inspection
            inspection = e.value == 'Inspect'
            dirty()

        def mouse(e: events.MouseEventArguments) -> None:
            nonlocal last_mouse
            if not inspection:
                return
            if e.type == 'mousedown':
                last_mouse = (e.image_x, e.image_y)
            elif e.type in {'mouseup', 'mouseleave'}:
                last_mouse = None
            elif last_mouse is not None:
                dx, dy = e.image_x-last_mouse[0], e.image_y-last_mouse[1]
                state.inspect_yaw -= dx*.25
                state.inspect_pitch = max(-89, min(89, state.inspect_pitch+dy*.25))
                last_mouse = (e.image_x, e.image_y)
                dirty()

        ui.label('FRC scene renderer').classes('text-2xl')
        ui.label('Meters • NWU coordinates • angles in degrees • calibrated exports')
        with ui.row().classes('w-full items-start'):
            with ui.column().classes('grow'):
                ui.toggle(['Inspect', 'Robot camera'], value='Inspect', on_change=change_view)
                image = ui.interactive_image(size=(960, 540), on_mouse=mouse,
                                             events=['mousedown', 'mousemove', 'mouseup', 'mouseleave']).classes('w-full')
                status = ui.label('Initializing renderer…')
                @ui.refreshable
                def inspection_controls() -> None:
                    vector_controls(state, 'inspect_target', ('Target X', 'Target Y', 'Target Z'))
                    ui.number('Inspection distance (zoom)', value=state.inspect_distance, min=.1,
                              on_change=lambda e: (setattr(state, 'inspect_distance', e.value), dirty()))
                inspection_controls()
                ui.button('Export calibrated PNG + settings', on_click=export)
            with ui.column().classes('w-96'):
                ui.upload(label='GLB / STL / combined camera JSON', on_upload=upload, auto_upload=True)
                model_path = ui.input('Server model path')

                def add(path: str, rotation: tuple[float, float, float] = (0, 0, 0)) -> None:
                    nonlocal selected
                    if Path(path).suffix.lower() not in {'.glb', '.stl'} or not Path(path).is_file():
                        ui.notify('Choose an existing GLB or STL', type='negative')
                        return
                    state.pieces.append(Piece(name=Path(path).stem, path=path, import_rotation=rotation))
                    selected = len(state.pieces)-1
                    objects.refresh()
                    dirty()

                ui.button('Add server model', on_click=lambda: add(model_path.value))
                def add_cone() -> None:
                    nonlocal selected
                    state.pieces.append(default_cone())
                    selected = len(state.pieces)-1
                    objects.refresh()
                    dirty()

                with ui.row():
                    ui.button('Add cone', on_click=add_cone)
                    ui.button('Add cube', on_click=lambda: add(str(Path(state.field_dir)/'model_1.glb'), (90, 0, 0)))

                @ui.refreshable
                def robot() -> None:
                    ui.label('Robot pose')
                    vector_controls(state, 'robot_xy_heading', ('Robot X', 'Robot Y', 'Heading'))
                    ui.switch('Lens distortion', value=state.distortion,
                              on_change=lambda e: (setattr(state, 'distortion', e.value), dirty()))
                robot()

                @ui.refreshable
                def objects() -> None:
                    nonlocal selected
                    if not state.pieces:
                        return
                    selected = min(selected, len(state.pieces)-1)

                    def select(e: events.ValueChangeEventArguments) -> None:
                        nonlocal selected
                        selected = int(e.value)
                        objects.refresh()

                    ui.select({i: p.name for i, p in enumerate(state.pieces)}, value=selected, on_change=select)
                    piece = state.pieces[selected]
                    vector_controls(piece, 'position', ('X', 'Y', 'Z'))
                    vector_controls(piece, 'rotation', ('Roll', 'Pitch', 'Yaw'))
                    vector_controls(piece, 'import_rotation', ('Import roll', 'Import pitch', 'Import yaw'))
                    vector_controls(piece, 'color', ('STL red', 'STL green', 'STL blue'))
                    ui.number('Scale to meters (STL: mm=.001, inch=.0254)', value=piece.scale, min=1e-8,
                              on_change=lambda e: (setattr(piece, 'scale', e.value), dirty()))

                    def duplicate() -> None:
                        state.pieces.append(piece.model_copy(deep=True))
                        objects.refresh()
                        dirty()

                    def remove() -> None:
                        state.pieces.pop(selected)
                        objects.refresh()
                        dirty()

                    with ui.row():
                        ui.button('Duplicate', on_click=duplicate)
                        ui.button('Remove', on_click=remove)
                        for label in ['Upright', 'Fallen']:
                            ui.button(label, on_click=lambda: ui.notify('Placement preset is a stub; pose unchanged'))
                objects()

                @ui.refreshable
                def lighting() -> None:
                    vector_controls(state, 'ambient', ('Ambient R', 'Ambient G', 'Ambient B'))
                    for index, light in enumerate(state.lights):
                        with ui.expansion(f'Light {index+1}: {light.kind}'):
                            ui.select(['directional', 'point', 'spot'], value=light.kind,
                                      on_change=lambda e, l=light: (setattr(l, 'kind', e.value), dirty()))
                            vector_controls(light, 'position', ('Light X', 'Light Y', 'Light Z'))
                            vector_controls(light, 'direction', ('Direction X', 'Direction Y', 'Direction Z'))
                            vector_controls(light, 'color', ('Red', 'Green', 'Blue'))
                            ui.number('Intensity', value=light.intensity, min=0,
                                      on_change=lambda e, l=light: (setattr(l, 'intensity', e.value), dirty()))
                            for field in ['inner_degrees', 'outer_degrees']:
                                ui.number(field, value=getattr(light, field), min=1, max=178,
                                          on_change=lambda e, l=light, f=field: (setattr(l, f, e.value), dirty()))
                            ui.switch('Shadows', value=light.shadow,
                                      on_change=lambda e, l=light: (setattr(l, 'shadow', e.value), dirty()))
                            def remove_light(l: Light = light) -> None:
                                state.lights.remove(l)
                                lighting.refresh()
                                dirty()
                            ui.button('Remove light', on_click=remove_light)
                    def add_light() -> None:
                        state.lights.append(Light(kind='point'))
                        lighting.refresh()
                        dirty()
                    ui.button('Add light', on_click=add_light)
                lighting()
                document = ui.input('Scene JSON path', value='runtime/scene.json')
                ui.button('Save scene + download', on_click=save_document)
                ui.button('Load server scene', on_click=load_document)

        async def tick() -> None:
            nonlocal busy, rendered_revision
            if busy or rendered_revision == revision:
                return
            busy = True
            current = revision
            started = time.monotonic()
            try:
                frame = await asyncio.get_running_loop().run_in_executor(
                    service.executor, service.render, state.model_copy(deep=True), True, inspection)
                buffer = io.BytesIO()
                Image.fromarray(frame).save(buffer, format='JPEG', quality=85)
                image.set_source('data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode())
                status.set_text(f'{time.monotonic()-started:.3f} s / frame • {frame.shape[1]}×{frame.shape[0]}')
            except Exception as error:
                status.set_text(f'Render failed: {error}. Check Vulkan and GPU driver availability.')
            finally:
                rendered_revision = current
                busy = False
        ui.timer(1/15, tick)
    ui.run(root=page, host=host, port=port, reload=False, show=False, title='FRC SAPIEN viewer')
