#!/usr/bin/env python3
"""Train an Ultralytics YOLO detector with resumable optimizer state.

Examples:
    python train_yolo.py
    python train_yolo.py --epochs 100 --name charged-up
    python train_yolo.py --resume runs/detect/charged-up/weights/final.pt --epochs 200

``--epochs`` is the total target epoch, not the number of additional epochs
when resuming.  Ultralytics restores the optimizer, scaler, and epoch from
the checkpoint instead of starting a fresh optimizer.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO


def save_verified_final_checkpoint(trainer) -> None:
    """Re-save and verify the final checkpoint before making a stable copy."""
    # Ultralytics normally saves last.pt after every epoch. Calling save_model
    # here also covers versions/configurations where the final save is skipped.
    trainer.save_model()

    last_path = Path(trainer.last)
    if not last_path.is_file():
        raise RuntimeError(f"Ultralytics did not create the final checkpoint: {last_path}")

    try:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        checkpoint = torch.load(last_path, map_location="cpu")

    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is None:
        raise RuntimeError(
            f"Final checkpoint {last_path} has no optimizer state; refusing to mark it resumable."
        )

    final_path = last_path.with_name("final.pt")
    shutil.copy2(last_path, final_path)
    print(
        f"Verified optimizer state in epoch {checkpoint.get('epoch')} and saved resumable checkpoint: "
        f"{final_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("raw_datasets/lincolnsudbury/data.yaml"),
        help="YOLO dataset YAML file",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="Initial model or weights file")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint to resume, e.g. runs/detect/charged-up/weights/final.pt",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Total target epoch")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="CUDA device, MPS, or cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="charged-up")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save a checkpoint every N epochs (Ultralytics save_period)",
    )
    parser.add_argument(
        "--strong-augment",
        action="store_true",
        help="Use stronger geometric, color, and multi-image augmentation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(f"Resume checkpoint does not exist: {args.resume}")
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {args.data}")

    # Ultralytics 8.4 restores ``epochs``, ``project``, and ``name`` from the
    # checkpoint and ignores overrides for those fields.  Use a temporary
    # checkpoint with the requested continuation settings so ``--epochs`` is
    # a real total target and the resumed run is written where requested.
    resume_tmp = None
    resume_path = args.resume
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint["train_args"] = dict(checkpoint.get("train_args", {}))
        checkpoint["train_args"].update(
            {
                "epochs": args.epochs,
                "project": str(args.project),
                "name": args.name,
                # Keep ``auto`` so Ultralytics reconstructs the saved MuSGD
                # optimizer for this long run and can load its 8 param groups.
                "optimizer": "auto",
            }
        )
        resume_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        resume_tmp.close()
        resume_path = Path(resume_tmp.name)
        torch.save(checkpoint, resume_path)

    model = YOLO(str(resume_path or args.model))
    model.add_callback("on_train_end", save_verified_final_checkpoint)

    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "patience": args.patience,
        "save": True,
        "save_period": args.save_period,
    }
    if args.strong_augment:
        # Broaden the training distribution while retaining a short clean-image
        # tail via close_mosaic.  These are intentionally train-only settings;
        # validation remains unaugmented.
        train_args.update(
            {
                "hsv_h": 0.02,
                "hsv_s": 0.9,
                "hsv_v": 0.5,
                "degrees": 8.0,
                "translate": 0.2,
                "scale": 0.7,
                "shear": 5.0,
                "perspective": 0.0005,
                "flipud": 0.05,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.15,
                "cutmix": 0.1,
                "copy_paste": 0.1,
                "close_mosaic": 15,
            }
        )
    if args.device is not None:
        train_args["device"] = args.device
    if args.resume is not None:
        train_args["resume"] = str(resume_path)
        # Ultralytics restores most arguments from the checkpoint on resume.
        # Keep the requested run destination and total epoch target so a
        # continuation can extend an older run instead of silently stopping at
        # the checkpoint's original epoch limit.
        train_args["project"] = str(args.project)
        train_args["name"] = args.name
        train_args["epochs"] = args.epochs
        # Let Ultralytics select the optimizer from the resumed run length;
        # for this combined run it reconstructs MuSGD, matching the checkpoint.
        train_args["optimizer"] = "auto"

    try:
        model.train(**train_args)
    finally:
        if resume_tmp is not None and resume_path is not None:
            resume_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
