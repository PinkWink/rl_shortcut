"""Road segmentation with YOLO: camera image -> road mask -> the 8 x 16 road view the policy was trained on.

Weights: the newest logs/pinky_yolo/*/weights/best.pt (scripts/pinky_yolo.py), else checkpoints/pinky_road_seg.pt
(not in the repository; the lecture page has it as an attachment).
Classes: 0 road (asphalt, centre line, crosswalk), 1 sidewalk (and the park). Everything else is background.
"""
import torch
import torch.nn.functional as F

from rl_shortcut import CHECKPOINT_DIR, LOG_DIR, pink_city

CLASSES = ["road", "sidewalk"]


WEIGHTS = None      # set by --weights (play.py, pinky_yolo.py): use this file instead of the newest training result


def seg_weights():
    if WEIGHTS is not None:
        return WEIGHTS
    runs = sorted(LOG_DIR.glob("pinky_yolo/*/weights/best.pt"), key=lambda p: p.parent.parent.name)
    if runs:
        return runs[-1]
    shipped = CHECKPOINT_DIR / "pinky_road_seg.pt"
    return shipped if shipped.exists() else None


def mask_to_view(mask):
    """Road mask (N, H, W) of the whole image -> road view (N, ROWS * COLS): the lower part, averaged into cells."""
    top = round(pink_city.VIEW_TOP * mask.shape[1])
    lower = mask[:, top:].float().unsqueeze(1)
    return F.adaptive_avg_pool2d(lower, (pink_city.VIEW_ROWS, pink_city.VIEW_COLS)).flatten(1)


class RoadSeg:
    def __init__(self, weights=None, conf=0.4):
        from ultralytics import YOLO
        self.weights = weights or seg_weights()
        if self.weights is None:
            raise SystemExit("no road segmentation weights: run scripts/pinky_yolo.py first")
        self.model, self.conf = YOLO(str(self.weights)), conf
        print(f"[yolo] {self.weights}")

    @torch.no_grad()
    def road_mask(self, rgb):
        """rgb (N, H, W, 3) uint8 on the GPU -> road mask (N, H, W) bool."""
        n, h, w, _ = rgb.shape
        x = rgb.permute(0, 3, 1, 2).float() / 255.0
        pad = (-h) % 32                                   # YOLO wants a multiple of 32: 360 -> 384
        x = F.pad(x, (0, 0, pad // 2, pad - pad // 2), value=114 / 255)
        results = self.model.predict(x, conf=self.conf, verbose=False)
        out = torch.zeros(n, h + pad, w, dtype=torch.bool, device=rgb.device)
        for i, r in enumerate(results):
            if r.masks is not None:
                road = r.masks.data[r.boxes.cls == 0]     # every road instance, merged
                if len(road):
                    out[i] = F.interpolate(road[None].float(), size=out.shape[1:]).amax(dim=1)[0] > 0.5
        return out[:, pad // 2: pad // 2 + h]


def yolo_view(env):
    """Observation term: road view from the robot camera (pinky_env.py, --perception yolo)."""
    if not hasattr(env, "_road_seg"):
        env._road_seg = RoadSeg()
    rgb = env.scene["camera"].data.output["rgb"][..., :3]
    mask = env._road_seg.road_mask(rgb)
    view = mask_to_view(mask)
    env._road_seg.last = {"rgb": rgb, "mask": mask, "view": view}     # for play.py --video
    return view


def debug_panel(env, viewport):
    """Viewport next to what robot 0 sees: camera with the YOLO road mask in green, and the 8 x 16 road view."""
    import numpy as np
    last = env._road_seg.last
    rgb, mask = last["rgb"][0].cpu().numpy(), last["mask"][0].cpu().numpy()
    view = last["view"][0].view(pink_city.VIEW_ROWS, pink_city.VIEW_COLS).cpu().numpy()
    seen = rgb.copy()
    seen[mask] = (0.45 * seen[mask] + 0.55 * np.array([80, 200, 80])).astype(np.uint8)
    h, w = rgb.shape[:2]
    cells = np.kron(view, np.ones((h // pink_city.VIEW_ROWS, w // pink_city.VIEW_COLS)))
    cells = np.pad(cells, ((0, h - cells.shape[0]), (0, w - cells.shape[1])), mode="edge")
    right = np.concatenate([seen, np.repeat((cells * 255).astype(np.uint8)[..., None], 3, axis=2)], axis=0)
    scale = right.shape[0] / viewport.shape[0]
    cols = (np.arange(int(viewport.shape[1] * scale)) / scale).astype(int)
    rows = (np.arange(right.shape[0]) / scale).astype(int)
    return np.concatenate([viewport[rows][:, cols, :3], right], axis=1)
