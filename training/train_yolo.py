"""Train a YOLO model using an Ultralytics training YAML file."""

import argparse
from pathlib import Path


class TrainingArgs(argparse.Namespace):
    config_path: str
    data: Path
    save_period: int
    name: str


def parse_args() -> TrainingArgs:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="YOLO dataset YAML file",
    )
    parser.add_argument(
        "config_path", help="Ultralytics training YAML file", required=True
    )
    parser.add_argument(
        "--save-period",
        type=int,
        required=True,
        default=-1,
        help="Save a checkpoint every N epochs (Ultralytics save_period)",
    )
    parser.add_argument("--name", default="charged-up")
    return parser.parse_args(namespace=TrainingArgs())


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO
    from ultralytics.utils import YAML

    config = YAML.load(args.config_path)
    model_path: object = config.get("model")
    if not isinstance(model_path, str) or not model_path.strip():
        raise SystemExit(
            f"Set 'model' to a model or weights path in {args.config_path}."
        )

    model = YOLO(model_path)
    model.train(
        cfg=args.config_path,
        data=str(args.data),
        save_period=args.save_period,
        name=args.name,
    )


if __name__ == "__main__":
    main()
