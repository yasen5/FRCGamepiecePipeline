# Charged Up YOLO Training

Install the training dependency in the project virtual environment:

```bash
python -m pip install -r requirements.txt
```

Start a run:

```bash
python train_yolo.py --epochs 100 --device 0
```

The script saves Ultralytics' rolling checkpoint at
`runs/detect/charged-up/weights/last.pt`. At training end it explicitly
re-saves and verifies the final checkpoint, then copies it to
`runs/detect/charged-up/weights/final.pt`. The checkpoint contains optimizer
state, so momentum/Adam statistics are preserved.

Resume to a new total epoch target:

```bash
python train_yolo.py \
  --resume runs/detect/charged-up/weights/final.pt \
  --epochs 200 \
  --device 0
```

Do not use `model.train()` with only a weights file for continuation; use
`--resume` so Ultralytics restores the optimizer and training epoch.
