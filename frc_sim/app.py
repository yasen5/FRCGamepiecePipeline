"""Single shared scene browser UI. Access through a localhost SSH tunnel."""
from __future__ import annotations

import asyncio
import base64
import io
import time
import math
from pathlib import Path
from uuid import uuid4

from nicegui import events, ui
from PIL import Image

from .model import Calibration, Light, Piece, Scene, GamepiecePlacement, default_cone, load_scene, scene_json
from .renderer import RenderService
from .interaction import place


def launch(state: Scene, host: str, port: int) -> None:
    service = RenderService()
    def page() -> None:
        nonlocal state
        busy = False
        revision = 0
        rendered_snapshot: Scene | None = None
        rendered_png: bytes | None = None
        dragging: str | None = None
        field_width = 16.54175
        field_height = 8.0137
        map_width, map_height = 1100, 533
        selected = 0

        def dirty() -> None:
            nonlocal revision
            revision += 1
            field_map.set_content(overlay())
            pose_label.set_text("Robot X %.2f m • Y %.2f m • heading %.1f°" % state.robot_xy_heading)
            if not busy:
                status.set_text("Scene changed — click Render")

        def set_vector(record: object, field: str, index: int, value: float | None) -> None:
            if value is None:
                return
            vector = list(getattr(record, field))
            vector[index] = float(value)
            setattr(record, field, tuple(vector))
            if isinstance(record, Piece) and record.gamepiece is not None and field == 'import_rotation':
                place(record, *record.position[:2])
            dirty()

        def vector_controls(record: object, field: str, labels: tuple[str, str, str]) -> None:
            with ui.row():
                for index, label in enumerate(labels):
                    with ui.column().classes('w-28'):
                        ui.label(label)
                        low, high, step = (-180., 180., 1.) if 'rotation' in field else ((0., 1., .01) if field in {'color', 'ambient'} else (-20., 20., .05))
                        ui.slider(min=low, max=high, step=step, value=getattr(record, field)[index],
                                  on_change=lambda e, j=index: set_vector(record, field, j, e.value)).props('label-always')

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
            nonlocal state, selected
            try:
                state = load_scene(Path(document.value))
                selected = 0
                objects.refresh()
                lighting.refresh()
                robot.refresh()
                quality.refresh()
                field_map.set_content(overlay())
                dirty()
                ui.timer(.01, load_background, once=True)
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

        def export() -> None:
            if rendered_png is None or rendered_snapshot is None:
                ui.notify('Render an image first', type='warning')
                return
            path = Path('runtime') / f'export-{uuid4().hex}.png'
            path.write_bytes(rendered_png)
            path.with_suffix('.json').write_text(scene_json(rendered_snapshot, path.with_suffix('.json')))
            ui.download.file(path)
            ui.download.file(path.with_suffix('.json'))

        def overlay() -> str:
            content = '<rect width="1100" height="533" fill="transparent" stroke="white"/>'
            for i, piece in enumerate(state.pieces):
                if piece.gamepiece is None:
                    continue
                x, y = piece.position[0]/field_width*map_width, (1-piece.position[1]/field_height)*map_height
                angle = math.radians(piece.gamepiece.theta)
                color = '#ffcc33' if i == selected else '#dddddd'
                content += f'<circle cx="{x}" cy="{y}" r="10" fill="{color}" stroke="black"/>'
                content += f'<line x1="{x}" y1="{y}" x2="{x+35*math.cos(angle)}" y2="{y-35*math.sin(angle)}" stroke="{color}" stroke-width="5"/>'
                content += f'<circle cx="{x+35*math.cos(angle)}" cy="{y-35*math.sin(angle)}" r="7" fill="{color}"/>'
            x, y, heading = state.robot_xy_heading
            px, py = x/field_width*map_width, (1-y/field_height)*map_height
            content += f'<circle cx="{px}" cy="{py}" r="15" fill="#44aaff"/>'
            angle = math.radians(heading)
            content += f'<line x1="{px}" y1="{py}" x2="{px+35*math.cos(angle)}" y2="{py-35*math.sin(angle)}" stroke="#44aaff" stroke-width="6"/>'
            content += f'<circle cx="{px+35*math.cos(angle)}" cy="{py-35*math.sin(angle)}" r="7" fill="#44aaff"/>'
            return content

        def mouse(e: events.MouseEventArguments) -> None:
            nonlocal dragging, selected
            x = max(0., min(field_width, e.image_x/map_width*field_width))
            y = max(0., min(field_height, (1-e.image_y/map_height)*field_height))
            if e.type == 'mousedown':
                dragging = None
                for i, piece in reversed(list(enumerate(state.pieces))):
                    if piece.gamepiece is None:
                        continue
                    px, py = piece.position[:2]
                    angle = math.radians(piece.gamepiece.theta)
                    if math.hypot(x-px-.53*math.cos(angle), y-py-.53*math.sin(angle)) < .2:
                        selected, dragging = i, 'theta'
                        break
                    if math.hypot(x-px, y-py) < .25:
                        selected, dragging = i, 'piece'
                        break
                if dragging is None:
                    rx, ry, heading = state.robot_xy_heading
                    if math.hypot(x-rx-.53*math.cos(math.radians(heading)), y-ry-.53*math.sin(math.radians(heading))) < .2:
                        dragging = 'robot_theta'
                    elif math.hypot(x-rx, y-ry) < .3:
                        dragging = 'robot'
                    else:
                        piece = default_cone() if asset.value == 'Cone' else Piece(name='Cube', path=str(Path(state.field_dir)/'model_1.glb'), import_rotation=(90, 0, 0), gamepiece=GamepiecePlacement())
                        if asset.value == 'Selected gamepiece':
                            if not state.pieces or state.pieces[selected].gamepiece is None:
                                ui.notify('Select a tagged gamepiece first', type='warning')
                                return
                            piece = state.pieces[selected].model_copy(deep=True)
                        state.pieces.append(piece)
                        selected = len(state.pieces)-1
                        dragging = 'piece'
                objects.refresh()
            if e.type in {'mouseup', 'mouseleave'}:
                dragging = None
            elif dragging is not None:
                try:
                    if dragging.startswith('robot'):
                        rx, ry, heading = state.robot_xy_heading
                        state.robot_xy_heading = (x, y, heading) if dragging == 'robot' else (rx, ry, math.degrees(math.atan2(y-ry, x-rx)))
                    else:
                        piece = state.pieces[selected]
                        if dragging == 'theta' and piece.gamepiece is not None:
                            piece.gamepiece.theta = math.degrees(math.atan2(y-piece.position[1], x-piece.position[0]))
                            x, y = piece.position[:2]
                        place(piece, x, y)
                    dirty()
                except Exception as error:
                    dragging = None
                    ui.notify(str(error), type='negative')
            field_map.set_content(overlay())

        ui.label('FRC scene renderer').classes('text-2xl')
        ui.label('Meters • NWU coordinates • angles in degrees • calibrated exports')
        with ui.row().classes('w-full items-start'):
            with ui.column().classes('grow'):
                ui.label('Drag markers to move; drag direction handles to rotate. Click the field to place a gamepiece.')
                field_map = ui.interactive_image(size=(map_width, map_height), on_mouse=mouse,
                    events=['mousedown', 'mousemove', 'mouseup', 'mouseleave']).classes('w-full bg-slate-800')
                field_map.set_content(overlay())
                pose_label = ui.label('Robot X %.2f m • Y %.2f m • heading %.1f°' % state.robot_xy_heading)
                asset = ui.select(['Cone', 'Cube', 'Selected gamepiece'], value='Cone', label='Click-to-place asset')
                render_button = ui.button('Render', on_click=lambda: render_image())
                status = ui.label('Click Render to create a ray-traced camera image')
                image = ui.image().classes('w-full')
                ui.button('Export calibrated PNG + settings', on_click=export)
            with ui.column().classes('w-96') as editor_panel:
                ui.upload(label='GLB / STL / combined camera JSON', on_upload=upload, auto_upload=True)
                model_path = ui.input('Server model path')

                def add(path: str, rotation: tuple[float, float, float] = (0, 0, 0)) -> None:
                    nonlocal selected
                    if Path(path).suffix.lower() not in {'.glb', '.stl'} or not Path(path).is_file():
                        ui.notify('Choose an existing GLB or STL', type='negative')
                        return
                    state.pieces.append(Piece(name=Path(path).stem, path=path, import_rotation=rotation, gamepiece=GamepiecePlacement() if Path(path).name == 'model_1.glb' else None))
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
                    if state.calibration is not None:
                        with ui.expansion('Camera mounting'):
                            mount = state.calibration.extrinsics
                            for field in ('translation_x', 'translation_y', 'translation_z', 'rotation_x', 'rotation_y', 'rotation_z'):
                                ui.label(field)
                                ui.slider(min=-180 if field.startswith('rotation') else -2, max=180 if field.startswith('rotation') else 2, step=.5 if field.startswith('rotation') else .01, value=getattr(mount, field), on_change=lambda e, f=field: (setattr(mount, f, e.value), dirty())).props('label-always')
                    ui.switch('Lens distortion', value=state.distortion,
                              on_change=lambda e: (setattr(state, 'distortion', e.value), dirty()))
                robot()
                @ui.refreshable
                def quality() -> None:
                    with ui.expansion('Render quality'):
                        ui.number('Samples per pixel', value=state.render_settings.samples_per_pixel, min=64, max=1024, step=64,
                              on_change=lambda e: (setattr(state.render_settings, 'samples_per_pixel', int(e.value)), dirty()) if e.value is not None else None)

                quality()

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
                    def update_placement() -> None:
                        try:
                            place(piece, *piece.position[:2])
                            field_map.set_content(overlay())
                            dirty()
                        except Exception as error:
                            ui.notify(str(error), type='negative')
                    def tag(e: events.ValueChangeEventArguments) -> None:
                        piece.gamepiece = GamepiecePlacement(theta=piece.rotation[2]) if e.value else None
                        objects.refresh()
                        field_map.set_content(overlay())
                        dirty()
                    ui.switch('Tagged gamepiece', value=piece.gamepiece is not None, on_change=tag)
                    if piece.gamepiece is not None:
                        placement = piece.gamepiece
                        ui.toggle({'upright': 'Upright', 'fallen': 'Down'}, value=placement.stance,
                                  on_change=lambda e: (setattr(placement, 'stance', e.value), update_placement()))
                        ui.label('Theta (degrees) — or drag the direction handle')
                        ui.slider(min=-180, max=180, step=1, value=placement.theta,
                                  on_change=lambda e: (setattr(placement, 'theta', float(e.value or 0)), update_placement())).props('label-always')
                        ui.label('Drag the piece on the field to change X/Y. Height is automatic.')
                    else:
                        ui.label('Tag as a gamepiece to edit X/Y, theta, and upright/down on the field.')

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

                objects()

                @ui.refreshable
                def lighting() -> None:
                    vector_controls(state, 'ambient', ('Ambient R', 'Ambient G', 'Ambient B'))
                    for index, light in enumerate(state.lights):
                        with ui.expansion(f'Light {index+1}: {light.kind}'):
                            ui.select(['directional', 'point', 'spot', 'area'], value=light.kind,
                                      on_change=lambda e, l=light: (setattr(l, 'kind', e.value), lighting.refresh(), dirty()))
                            vector_controls(light, 'position', ('Light X', 'Light Y', 'Light Z'))
                            vector_controls(light, 'direction', ('Direction X', 'Direction Y', 'Direction Z'))
                            vector_controls(light, 'color', ('Red', 'Green', 'Blue'))
                            ui.label('Intensity')
                            ui.slider(value=light.intensity, min=0, max=100, step=.05,
                                      on_change=lambda e, l=light: (setattr(l, 'intensity', e.value), dirty()))
                            if light.kind == 'area':
                                for dimension in ('width', 'height'):
                                    ui.number(dimension + ' (m)', value=getattr(light, dimension), min=.01,
                                              on_change=lambda e, l=light, f=dimension: (setattr(l, f, e.value), dirty()))
                            for field in ['inner_degrees', 'outer_degrees']:
                                ui.label(field)
                                ui.slider(value=getattr(light, field), min=1, max=178, step=1,
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

        async def load_background() -> None:
            try:
                frame = await asyncio.get_running_loop().run_in_executor(service.executor, service.field_map, state.model_copy(deep=True), map_width, map_height)
                buffer = io.BytesIO()
                Image.fromarray(frame).save(buffer, format='PNG')
                field_map.set_source('data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())
            except Exception as error:
                ui.notify(f'Field background failed: {error}', type='negative')

        async def render_image() -> None:
            nonlocal busy, rendered_snapshot, rendered_png
            if busy:
                return
            busy = True
            render_button.disable()
            current = revision
            snapshot = state.model_copy(deep=True)
            status.set_text('Ray tracing…')
            started = time.monotonic()
            try:
                frame = await asyncio.get_running_loop().run_in_executor(service.executor, service.render, snapshot)
                buffer = io.BytesIO()
                Image.fromarray(frame).save(buffer, format='PNG')
                png = buffer.getvalue()
                image.set_source('data:image/png;base64,' + base64.b64encode(png).decode())
                rendered_snapshot, rendered_png = snapshot, png
                stale = ' • Scene changed — click Render again' if revision != current else ''
                status.set_text(f'{time.monotonic()-started:.2f} s • {frame.shape[1]}×{frame.shape[0]} • {snapshot.render_settings.samples_per_pixel} samples/pixel' + stale)
            except Exception as error:
                status.set_text(f'Render failed: {error}. Check Vulkan ray-tracing support and GPU drivers.')
            finally:
                busy = False
                render_button.enable()

        ui.timer(.1, load_background, once=True)
    ui.run(root=page, host=host, port=port, reload=False, show=False, title='FRC SAPIEN viewer')
