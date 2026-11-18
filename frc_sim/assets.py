"""Pinned field acquisition and project-local storage links."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://github.com/Mechanical-Advantage/AdvantageScopeAssets/releases/download/archive-v1/Field3d_2023FRCFieldV5.zip'
SHA256 = 'ffd6141541d280380b9749065b197edbcd920fe1f2b1f2853adb8df9aa4015f9'


def fetch_field() -> Path:
    destination = Path('/mnt/ssd2/data/frc_sim/2023-field-v5')
    if not (destination / 'model.glb').exists():
        with urllib.request.urlopen(URL) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError('Field archive checksum mismatch')
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for name in ('config.json', 'model.glb', 'model_0.glb', 'model_1.glb'):
                (destination / name).write_bytes(archive.read(name))
        with urllib.request.urlopen('https://raw.githubusercontent.com/Mechanical-Advantage/AdvantageScopeAssets/main/LICENSE') as response:
            (destination / 'LICENSE').write_bytes(response.read())
        (destination / 'SOURCE.txt').write_text(f'{URL}\nsha256: {SHA256}\n')
    link = Path('assets/field')
    link.parent.mkdir(exist_ok=True)
    if not link.exists():
        link.symlink_to(destination, target_is_directory=True)
    return link
