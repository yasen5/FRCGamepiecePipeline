#!/usr/bin/env python3
"""Annotate ground-truth YOLO labels onto sample images from a dataset and
save them as a single panel image to disk.

Usage:
    python visualize_gt.py raw_datasets/cone_vision
    python visualize_gt.py raw_datasets/cone_vision --split valid --num-images 12 --output panel.jpg
"""
import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

BOX_COLORS = [
    (255, 99, 71),   # tomato
    (65, 105, 225),  # royal blue
    (60, 179, 113),  # medium sea green
    (255, 165, 0),   # orange
    (186, 85, 211),  # medium orchid
    (0, 206, 209),   # dark turquoise
]


def load_class_names(dataset_dir: Path) -> list[str]:
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["names"]


def load_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cls_idx, cx, cy, w, h = line.split()[:5]
        cls_idx = int(cls_idx)
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((cls_idx, x1, y1, x2, y2))
    return boxes


def annotate_image(img_path: Path, label_path: Path, class_names: list[str], font) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes = load_yolo_boxes(label_path, img.width, img.height)
    for cls_idx, x1, y1, x2, y2 in boxes:
        color = BOX_COLORS[cls_idx % len(BOX_COLORS)]
        line_width = max(2, round(min(img.width, img.height) * 0.004))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad = 3
        tag_y1 = max(0, y1 - text_h - 2 * pad)
        draw.rectangle([x1, tag_y1, x1 + text_w + 2 * pad, tag_y1 + text_h + 2 * pad], fill=color)
        draw.text((x1 + pad, tag_y1 + pad), label, fill=(255, 255, 255), font=font)
    return img


def build_panel(images: list[Image.Image], titles: list[str], cols: int, cell_size: int, font) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    title_h = 22
    cell_w = cell_h = cell_size
    panel = Image.new("RGB", (cols * cell_w, rows * (cell_h + title_h)), (30, 30, 30))
    draw = ImageDraw.Draw(panel)
    max_title_w = cell_w - 8
    for i, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(i, cols)
        thumb = img.copy()
        thumb.thumbnail((cell_w, cell_h))
        x_off = c * cell_w + (cell_w - thumb.width) // 2
        y_off = r * (cell_h + title_h) + title_h
        panel.paste(thumb, (x_off, y_off))
        while title and draw.textlength(title, font=font) > max_title_w:
            title = title[:-1]
        draw.text((c * cell_w + 4, r * (cell_h + title_h) + 4), title, fill=(255, 255, 255), font=font)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Path to a YOLO-format dataset (contains data.yaml)")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--num-images", type=int, default=9)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--cell-size", type=int, default=400, help="Thumbnail size in pixels per panel cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    class_names = load_class_names(dataset_dir)

    images_dir = dataset_dir / args.split / "images"
    labels_dir = dataset_dir / args.split / "labels"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    random.seed(args.seed)
    sample = random.sample(image_paths, min(args.num_images, len(image_paths)))

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    annotated, titles = [], []
    for img_path in sample:
        label_path = labels_dir / (img_path.stem + ".txt")
        annotated.append(annotate_image(img_path, label_path, class_names, font))
        titles.append(img_path.name)

    panel = build_panel(annotated, titles, args.cols, args.cell_size, font)

    output = args.output
    if output is None:
        output = Path(f"{dataset_dir.name}_{args.split}_gt_panel.jpg")
    panel.save(output, quality=92)
    print(f"Saved panel of {len(sample)} images from {dataset_dir}/{args.split} -> {output}")


if __name__ == "__main__":
    main()
