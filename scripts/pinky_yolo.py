"""Train YOLO segmentation to find the road in Pinky Pro camera images (Isaac Sim is not started).

  python scripts/pinky_yolo.py                      # train on datasets/pinky_road -> logs/pinky_yolo/<date_time>/
  python scripts/pinky_yolo.py --epochs 10          # shorter
  python scripts/pinky_yolo.py --check              # no training: run the newest weights on the validation images

Starts from yolo11n-seg.pt (downloaded on first use). Classes: 0 road, 1 sidewalk.
"""
import argparse
from datetime import datetime

import cv2
import numpy as np
import torch

from rl_shortcut import LOG_DIR, ROOT
from rl_shortcut.road_seg import RoadSeg, mask_to_view

p = argparse.ArgumentParser()
p.add_argument("--data", default="datasets/pinky_road")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--check", action="store_true")
args = p.parse_args()


def train():
    from ultralytics import YOLO
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model = YOLO("yolo11n-seg.pt")
    result = model.train(data=str(ROOT / args.data / "data.yaml"), epochs=args.epochs, imgsz=640, batch=16, workers=4,
                         project=str(LOG_DIR / "pinky_yolo"), name=name, flipud=0.0, plots=True, verbose=False)
    m = result.results_dict
    print(f"[yolo] mask mAP50 {m['metrics/mAP50(M)']:.3f}, mAP50-95 {m['metrics/mAP50-95(M)']:.3f}")
    print(f"[yolo] weights: {LOG_DIR / 'pinky_yolo' / name / 'weights/best.pt'}")


def check(count=6):
    """Validation images: camera | YOLO road mask | the 8 x 16 road view the policy gets."""
    seg = RoadSeg()
    files = sorted((ROOT / args.data / "images/val").glob("*.jpg"))[:count]
    rows, ious = [], []
    for f in files:
        image = cv2.imread(str(f))
        rgb = torch.from_numpy(image[:, :, ::-1].copy()).cuda()[None]
        mask = seg.road_mask(rgb)
        view = mask_to_view(mask)[0].view(8, 16).cpu().numpy()
        # compare with the label of this image
        truth = np.zeros(image.shape[:2], np.uint8)
        for line in (ROOT / args.data / "labels/val" / (f.stem + ".txt")).read_text().splitlines():
            v = line.split()
            if v and v[0] == "0":
                pts = (np.array(v[1:], float).reshape(-1, 2) * image.shape[1::-1]).astype(np.int32)
                cv2.fillPoly(truth, [pts], 1)
        m = mask[0].cpu().numpy()
        ious.append((m & (truth > 0)).sum() / max((m | (truth > 0)).sum(), 1))
        overlay = image.copy()
        overlay[m] = (0.4 * overlay[m] + 0.6 * np.array([80, 200, 80])).astype(np.uint8)
        grid = cv2.resize((view * 255).astype(np.uint8), image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        rows.append(np.concatenate([image, overlay, cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)], axis=1))
    out = ROOT / "outputs/pinky_yolo_check.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(rows, axis=0))
    print(f"[yolo] road IoU on {len(files)} validation images: {np.mean(ious):.3f}")
    print(f"[yolo] saved {out}  (camera | road mask | road view 8 x 16)")


if __name__ == "__main__":
    check() if args.check else train()
