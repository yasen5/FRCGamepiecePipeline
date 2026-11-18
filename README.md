# Calibrated FRC scene renderer

A Python SAPIEN renderer with a browser control panel for the 2023 CHARGED UP field. Prefer **GLB**, which includes materials and textures. STL is supported with a configurable solid material; convert STEP to GLB in your CAD application or Blender first.

## Setup

Run from this directory. Python 3.11+, a working Vulkan-capable GPU driver, and network access for the initial field download are required. No display server is needed.

Per this project's environment policy, **you must create the environment symlink** before installation:

```sh
ln -s ../.venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
./.venv/bin/python -m frc_sim fetch-field
./.venv/bin/python -m frc_sim view
```

The existing environment is shared with the parent project. The application binds to `127.0.0.1:8080`. On your own computer, forward the server port:

```sh
ssh -L 8080:127.0.0.1:8080 USER@SERVER
```

Then open <http://localhost:8080>. All connected browsers control the same scene. Keep this service behind the SSH tunnel; it intentionally provides server-file loading for its owner.

The observed NVIDIA `Driver/library version mismatch` must be repaired on the host before GPU validation. A reboot after a driver update may resolve it; confirm a functioning Vulkan device as well as `nvidia-smi`. The renderer reports initialization errors in the preview status.

## Default gamepiece

New scenes and **Add cone** use `2023ChargedUpConeChopped.glb` from the repository root. It remains at its original scale (meters). The import preset rotates it 180° about X and sets its origin 0.0127 m above the carpet so its base sits at Z=0. The model file is unchanged. This import correction is independent of the upright/fallen stub buttons. Existing saved scenes retain their explicit asset references.

## Using the viewer

1. Start in **Inspect** to see the field. Drag the image to orbit; change the target coordinates to pan and distance to zoom.
2. Add your chopped cone or the provided cube, upload GLB/STL files, or enter an existing server model path. Select an instance to edit its position and roll/pitch/yaw. Duplicate/remove controls act on the selected instance.
3. GLB units default to meters; its node hierarchy is retained. For STL exported in millimeters set scale to `0.001`; for inches use `0.0254`. Import rotations correct source axes before applying the object's scene rotation. Position is the imported model origin, not an automatically inferred ground contact. Nothing is settled by physics.
4. Upload a combined camera JSON like `frc971/cos/constants/vision-bot/front.json`. Choose **Robot camera**, then edit robot X/Y/heading. Camera mounting remains fixed relative to the robot.
5. Adjust ambient light and add/edit directional, point, or spot lights. Directions must be nonzero; spot inner angles must not exceed outer angles. Shadows use rasterized shadow maps, not ray tracing.
6. Export a calibrated PNG and accompanying scene JSON. Inspection view changes do not affect this export. Save/load scenes using server JSON paths; move associated assets with their relative references if transferring a scene elsewhere.

Upright/fallen buttons are stubs and display a notification without changing placement. Manual rotation is available. Preview images are compressed JPEG; exported RGB PNGs are lossless and use the calibration's original dimensions. Changes request previews at up to 15 Hz; actual speed depends on GPU, model complexity, transfer time, and shadow lighting. No idle frames are rendered.

## Calibration and coordinate conventions

Accepts **one combined JSON**, with acquisition fields ignored:

```json
{
  "width": 1280, "height": 800,
  "extrinsics": {
    "translation_x": 0.2, "translation_y": 0, "translation_z": 0.3,
    "rotation_x": 0, "rotation_y": -20, "rotation_z": 0
  },
  "intrinsics": {
    "fx": 900, "fy": 910, "cx": 650, "cy": 403,
    "k1": 0.04, "k2": -0.1, "k3": 0.05, "p1": 0.001, "p2": -0.002
  }
}
```

Mount translation is meters, rotation is degrees. It describes the camera pose **in the robot frame**, not an already inverted world-to-camera matrix. Rotation is `Rz(yaw) @ Ry(pitch) @ Rx(roll)`. Camera/robot axes are X forward, Y left, Z up; negative pitch points upward. OpenCV optical axes are handled through SAPIEN's calibrated projection and the OpenCV distortion map.

World coordinates use WPILib's blue-origin field system: X toward red, Y left when facing red, Z up. The field's configured import rotation, 180-degree centered-field correction, and half-field translation are applied once. Robot Z/roll/pitch remain zero in this version. Intrinsics allow unequal focal lengths and off-center optical centers. Preview scales X/Y intrinsics with their actual rounded image dimensions. Distortion defaults on; disable it to obtain a pinhole image. `k1/k2/k3/p1/p2` default to zero when absent.

Example calibration source: https://github.com/frc971/cos/blob/main/constants/vision-bot/front.json
Transform interpretation: https://github.com/frc971/cos/blob/main/src/utils/camera_config.cc

## Reproducible single-image rendering

Save a scene in the browser, then render without starting the browser server:

```sh
./.venv/bin/python -m frc_sim render --scene runtime/scene.json --output runtime/example.png
```

JSON asset paths are relative to the JSON file. Exports embed the effective calibration, object transforms, lights, robot pose, and distortion setting. The matching PNG and JSON share a basename. Rendering neither simulates exposure nor copies acquisition FPS into a physical simulation. Upright/fallen settling, physics, batch datasets, depth/masks, and STEP conversion are not implemented.

`assets/field` links to `/mnt/ssd2/data/frc_sim/2023-field-v5`. A new `runtime` link points to a unique folder under `/mnt/ssd2/tmp_outputs/` on first launch. Uploads, cleaned field caches, and default exports go there. Keep the target folder if you want its saved scenes to remain usable; downloads do not bundle referenced assets.

The pinned [AdvantageScope field archive](https://github.com/Mechanical-Advantage/AdvantageScopeAssets/releases/download/archive-v1/Field3d_2023FRCFieldV5.zip) is verified with SHA-256 `ffd6141541d280380b9749065b197edbcd920fe1f2b1f2853adb8df9aa4015f9`. Its license and source attribution accompany the local asset. Staged gamepiece nodes are omitted from the field rendering; added instances are separate.

## Validation

```sh
./.venv/bin/python -m pytest -q
./.venv/bin/pyright --pythonpath ./.venv/bin/python
FRC_GPU_TEST=1 ./.venv/bin/python -m pytest tests/test_render.py -q
```

CPU tests exercise mount composition, scaled calibrated projection, inverse/forward lens mapping, validation, and scene save/load. The explicit GPU test loads the actual field and both gamepieces, exports and reloads, checks dimensions and repeatability, then changes poses and lighting. GPU failures are failures when this test is enabled. Browser uploads, mouse controls, downloads, and shadow appearance also require a manual browser check on a working GPU server.
