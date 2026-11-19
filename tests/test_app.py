"""Field editing to explicit PNG download, without requiring a GPU."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from nicegui import ui
import pytest

from frc_sim import app
from frc_sim.model import Scene, load_scene


def test_explicit_render_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    state = Scene()
    calls: list[Scene] = []
    downloads: list[Path] = []
    actions: dict[str, Callable] = {}
    fail = False
    executor = ThreadPoolExecutor(max_workers=1)

    class ControlledService:
        def __init__(self) -> None:
            self.executor = executor

        def render(self, snapshot: Scene) -> NDArray[np.uint8]:
            if fail:
                raise RuntimeError("controlled render failure")
            calls.append(snapshot)
            return np.full((800, 1280, 3), 80, dtype=np.uint8)

    original_button = ui.button

    def button(text: str, *, on_click: Callable, **kwargs: object) -> ui.button:
        actions[text] = on_click
        return original_button(text, on_click=on_click)

    async def exercise(root: Callable[[], None]) -> None:
        nonlocal fail
        root()
        assert not calls
        state.robot_xy_heading = (3, 4, 90)
        await actions['Render']()
        assert len(calls) == 1
        assert calls[0].robot_xy_heading == (3, 4, 90)
        state.robot_xy_heading = (5, 4, 0)
        actions['Export calibrated PNG + settings']()
        assert len(calls) == 1
        from PIL import Image
        assert Image.open(downloads[0]).size == (1280, 800)
        assert load_scene(downloads[1]).robot_xy_heading == (3, 4, 90)
        fail = True
        await actions['Render']()
        actions['Export calibrated PNG + settings']()
        assert len(calls) == 1
        assert downloads[0].read_bytes() == downloads[2].read_bytes()
        assert load_scene(downloads[3]).robot_xy_heading == (3, 4, 90)
        fail = False
        await actions['Render']()
        assert len(calls) == 2

    monkeypatch.setattr(app, 'RenderService', ControlledService)
    monkeypatch.setattr(ui, 'button', button)
    monkeypatch.setattr(ui, 'timer', lambda *args, **kwargs: None)
    monkeypatch.setattr(ui.download, 'file', lambda path: downloads.append(Path(path)))
    monkeypatch.setattr(ui, 'run', lambda *, root, **kwargs: asyncio.run(exercise(root)))
    try:
        app.launch(state, '127.0.0.1', 8080)
    finally:
        executor.shutdown()
        for path in downloads:
            path.unlink(missing_ok=True)
