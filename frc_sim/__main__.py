from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from .model import Scene, load_scene


def main() -> None:
    parser = argparse.ArgumentParser(description='Headless SAPIEN FRC viewer and PNG renderer')
    parser.add_argument('command', choices=['fetch-field', 'view', 'render'])
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runtime/render.png'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    if args.command == 'fetch-field':
        from .assets import fetch_field
        print(fetch_field())
        return
    runtime = Path('runtime')
    if not runtime.exists():
        target = Path('/mnt/ssd2/tmp_outputs') / f'frc-sim-{uuid4().hex}'
        target.mkdir(parents=True)
        runtime.symlink_to(target, target_is_directory=True)
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
