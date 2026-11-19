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

After a driver update, an NVIDIA `Driver/library version mismatch` may require rebooting the host. Confirm a functioning Vulkan device as well as `nvidia-smi`. Headless raster and ray-tracing validation passed on the RTX 4070 with driver 595.99.02 after reboot. The renderer reports initialization errors in the render status.

## Default gamepiece

New scenes and **Add cone** use `2023ChargedUpConeChopped.glb` from the repository root. It remains at its original scale (meters). The import preset rotates it 180° about X and sets its origin 0.0127 m above the carpet so its base sits at Z=0. The model file is unchanged. This import correction is independent of the upright/fallen stub buttons. Existing saved scenes retain their explicit asset references.

## Using the viewer

1. Use the top-down field to place gamepieces and drag robot or gamepiece markers. Drag a direction handle to change theta (counterclockwise from field +X).
2. Choose Cone, Cube, or Selected gamepiece to place on the carpet. Choose Upright or Down for a selected gamepiece; height is calculated from mesh contact.
3. Click **Render** to create one ray-traced camera image. Editing never triggers camera rendering. The previous image stays visible and is marked outdated after changes.
4. Camera output uses the calibration's native resolution (1280×800 by default), transmitted as lossless PNG. Ray tracing defaults to 256 samples per pixel, 8 bounces, and no denoising. Change samples under Render quality (64–1024).
5. New scenes use overhead area lights and weak ambient fill. Existing saved scenes retain explicit lighting. Edit light positions, directions, colors, intensity, and area dimensions in the controls. Final lighting intensity needs visual tuning on a working GPU.
6. Upload GLB/STL models or a combined camera JSON, or enter server paths. Tag imported models as gamepieces to enable field placement. Camera mounting and lighting have their own controls.
7. **Export calibrated PNG + settings** downloads the displayed image and its exact scene snapshot, even if you have since edited the scene. Save/load scene controls handle the current editable scene independently.

The field background is a cached orthographic raster image with editor illumination. Robot camera output, exports, and CLI renders use SAPIEN's ray-tracing shader and require compatible Vulkan ray-tracing hardware and drivers. Failures are reported without substituting raster output. All connected browsers share the scene; each browser requests its own stills.

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

JSON asset paths are relative to the JSON file. Exports embed the effective calibration, object transforms, lights, robot pose, and distortion setting. The matching PNG and JSON share a basename. Rendering neither simulates exposure nor copies acquisition FPS into a physical simulation. Physics settling, depth/masks, and STEP conversion are not implemented.

`assets/field` links to `/mnt/ssd2/data/frc_sim/2023-field-v5`. A new `runtime` link points to a unique folder under `/mnt/ssd2/tmp_outputs/` on first launch. Uploads, cleaned field caches, and default exports go there. Keep the target folder if you want its saved scenes to remain usable; downloads do not bundle referenced assets.

The pinned [AdvantageScope field archive](https://github.com/Mechanical-Advantage/AdvantageScopeAssets/releases/download/archive-v1/Field3d_2023FRCFieldV5.zip) is verified with SHA-256 `ffd6141541d280380b9749065b197edbcd920fe1f2b1f2853adb8df9aa4015f9`. Its license and source attribution accompany the local asset. Staged gamepiece nodes are omitted from the field rendering; added instances are separate.

## Generate batched gamepiece layouts

From the repository root, with the field assets available, run:

```sh
./.venv/bin/python -m frc_sim generate-layouts --batch-size 32 --num-batches 10 --seed 2023
```

The command prints the run directory under `runtime/`. Use `--output` to name a new directory. Each numbered JSON file contains one gamepiece layout and exactly `--batch-size` robot poses. Reuse the layout across those views and compose each robot pose with the camera extrinsics.

```json
{
  "version": 1,
  "gamepieces": [
    {"kind": "cone", "position": [4.0, 5.0, 0.0127], "rotation": [0, 0, 130]}
  ],
  "robot_poses": [[5.2, 4.1, 135], [6.3, 4.7, 180]]
}
```

Positions are in meters in the blue-origin field frame. Gamepiece angles are roll, pitch, and yaw in degrees before applying the asset's import correction. A cone uses the chopped cone with its 180° X import correction. A cube uses `assets/field/model_1.glb` with a 90° X import correction. Robot poses are `[x, y, theta]`, with counterclockwise heading in degrees and implicit zero Z, roll, and pitch. A downstream renderer supplies the assets and calibration.

The piece count is a rounded normal draw clamped to zero, controlled by `--piece-count-mean` (8) and `--piece-count-std` (3). Piece types and upright/fallen stances are sampled independently with equal probability. The robot footprint defaults to 0.8 × 0.8 m including bumpers. Set `--robot-length` and `--robot-width` for a different robot. Poses must clear the gamepieces, field boundary, and conservative static obstacle bounds. The first run caches field bounds and piece contact heights under `runtime/`. Subsequent runs reuse that cache without loading meshes.

To require at least two pieces in every left COS camera view and render the resulting layouts:

```sh
./.venv/bin/python -m frc_sim generate-layouts --batch-size 5 --num-batches 10 --min-visible 2 --seed 2023 --output runtime/cos-left-layouts
./.venv/bin/python -m frc_sim render-layouts --layouts runtime/cos-left-layouts --output runtime/cos-left-renders
```

`--min-visible` defaults to zero. For a gamepiece to count, a strict majority of its points must project inside the calibrated left-camera image and clear the conservative occlusion bounds of field objects and other gamepieces. By default each gamepiece has one point at the center of its mesh-local bounding box. The renderer writes `batch_0/layout.json`, `batch_0/frame_0.png`, and matching scene JSON sidecars, with one numbered directory per layout.

## Keypoint annotations

Batch rendering writes one `keypoints.json` at the root of the output directory. Standalone `render` writes `<image-stem>.keypoints.json` beside the image. The versioned manifest lists the class names in `classes` and one record per frame in `images`. Each frame has a relative `image` path, `width`, `height`, and a `points` list. Each point has a zero-based `class_id` into `classes`, a zero-based `piece_index` into the scene sidecar's `pieces` array, and fractional pixel coordinates `x`, `y` measured from the upper-left. The manifest includes in-frame projections of occluded points. A frame with no in-frame points has an empty list.

By default the only class is `center`. To use named landmarks, save the following example as `runtime/landmarks.json` and pass it explicitly to both layout commands. Positions are in asset-local meters after applying any mesh node transforms and before applying the scene's scale, import rotation, or field pose. The cone's `top` is the center of its chopped tip. Each rounded corner is represented by a mesh point.

```json
{
  "cone": [
    {"name": "top", "position": [0, 0, -0.31273749]},
    {"name": "corner", "position": [0.0984622, 0.0980155, 0.0127]},
    {"name": "corner", "position": [0.0984622, -0.0980155, 0.0127]},
    {"name": "corner", "position": [-0.0984622, -0.0980155, 0.0127]},
    {"name": "corner", "position": [-0.0984622, 0.0980155, 0.0127]}
  ],
  "cube": [
    {"name": "corner", "position": [-0.09181478, -0.08889998, -0.09123633]},
    {"name": "corner", "position": [-0.09123633, -0.08889998, 0.09181478]},
    {"name": "corner", "position": [-0.09181478, 0.08889998, -0.09123633]},
    {"name": "corner", "position": [-0.09123633, 0.08889998, 0.09181478]},
    {"name": "corner", "position": [0.09123633, -0.08889998, -0.09181478]},
    {"name": "corner", "position": [0.09181478, -0.08889998, 0.09123633]},
    {"name": "corner", "position": [0.09123633, 0.08889998, -0.09181478]},
    {"name": "corner", "position": [0.09181478, 0.08889998, 0.09123633]}
  ]
}
```

```sh
./.venv/bin/python -m frc_sim generate-layouts --batch-size 5 --min-visible 2 --seed 2023 --output runtime/landmark-layouts --keypoints --keypoints-file runtime/landmarks.json
./.venv/bin/python -m frc_sim render-layouts --layouts runtime/landmark-layouts --output runtime/landmark-renders --keypoints --keypoints-file runtime/landmarks.json
```

With these definitions, three cone points or five cube points must pass the visibility check for the piece to count. For standalone rendering, put the corresponding landmark list in each gamepiece's `keypoints` field in the scene JSON, then pass `--keypoints` to `render`.

## Validation

```sh
./.venv/bin/python -m pytest -q
./.venv/bin/pyright --pythonpath ./.venv/bin/python
FRC_GPU_TEST=1 ./.venv/bin/python -m pytest tests/test_render.py -q
```

CPU tests exercise mount composition, scaled calibrated projection, inverse/forward lens mapping, validation, and scene save/load. The explicit GPU test loads the actual field and both gamepieces, exports and reloads, checks dimensions and repeatability, then changes poses and lighting. GPU failures are failures when this test is enabled. Browser uploads, mouse controls, downloads, and shadow appearance also require a manual browser check on a working GPU server.
