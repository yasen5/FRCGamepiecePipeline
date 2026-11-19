from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from .model import Scene, load_scene


def _runtime() -> Path:
    runtime = Path('runtime')
    if not runtime.exists():
        target = Path('/mnt/ssd2/tmp_outputs') / f'frc-sim-{uuid4().hex}'
        target.mkdir(parents=True)
        try:
            runtime.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            target.rmdir()
            if not runtime.is_dir():
                raise
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description='FRC field viewer, renderer, and layout generator')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('fetch-field', help='Fetch the pinned field assets')
    view = commands.add_parser('view', help='Launch the browser viewer')
    view.add_argument('--scene', type=Path)
    view.add_argument('--host', default='127.0.0.1')
    view.add_argument('--port', type=int, default=8080)
    render = commands.add_parser('render', help='Render one scene')
    render.add_argument('--scene', type=Path)
    render.add_argument('--output', type=Path, default=Path('runtime/render.png'))
    generate = commands.add_parser('generate-layouts', help='Generate one JSON layout per batch')
    generate.add_argument('--batch-size', type=int, required=True, help='Robot poses per layout')
    generate.add_argument('--num-batches', type=int, default=1)
    generate.add_argument('--piece-count-mean', type=float, default=8)
    generate.add_argument('--piece-count-std', type=float, default=3)
    generate.add_argument('--robot-length', type=float, default=.8, help='Bumper-to-bumper length in meters')
    generate.add_argument('--robot-width', type=float, default=.8, help='Bumper-to-bumper width in meters')
    generate.add_argument('--seed', type=int)
    generate.add_argument('--min-visible', '--min_visible', type=int, default=0,
                          help='Minimum unobstructed gamepieces in every left-camera view')
    generate.add_argument('--output', type=Path, help='New output directory (default: fresh directory under runtime)')
    batch_render = commands.add_parser('render-layouts', help='Render all robot poses in generated layouts')
    batch_render.add_argument('--layouts', type=Path, required=True)
    batch_render.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fetch-field':
        from .assets import fetch_field
        print(fetch_field())
        return
    if args.command == 'generate-layouts':
        from .layouts import LayoutOptions, generate_layouts, load_geometry

        try:
            options = LayoutOptions(batch_size=args.batch_size, num_batches=args.num_batches,
                                    piece_count_mean=args.piece_count_mean, piece_count_std=args.piece_count_std,
                                    robot_length=args.robot_length, robot_width=args.robot_width, seed=args.seed,
                                    min_visible=args.min_visible)
            runtime = _runtime()
            destination = args.output or runtime / f'gamepiece-layouts-{uuid4().hex}'
            print(generate_layouts(options, load_geometry(runtime), destination))
        except (OSError, ValueError) as error:
            generate.error(str(error))
        return
    _runtime()
    if args.command == 'render-layouts':
        from .batch_render import render_layouts
        try:
            print(render_layouts(args.layouts, args.output))
        except (OSError, ValueError, RuntimeError) as error:
            batch_render.error(str(error))
        return
    state = load_scene(args.scene) if args.scene else Scene()
    if args.command == 'view':
        from .app import launch
        launch(state, args.host, args.port)
    else:
        from .renderer import RenderService
        service = RenderService()
        try:
            print(service.executor.submit(service.export, state, args.output).result())
        finally:
            service.executor.shutdown()


if __name__ == '__main__':
    main()
