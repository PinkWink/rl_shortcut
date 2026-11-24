"""Collect a road segmentation dataset in the City of Pink: Pinky Pro camera images + labels, made automatically.

  python scripts/pinky_dataset.py --headless --enable_cameras              # 400 images -> datasets/pinky_road/
  python scripts/pinky_dataset.py --headless --enable_cameras --count 40 --out datasets/pinky_road_test

The label needs no hand painting: the camera also gives the distance of every pixel, so we know which floor point
a pixel shows, and the floor label map (assets/pink_city/textures/floor_labels.png) says what is painted there.
Pixels that are not floor (walls, buildings, trees, sky) are background.
Output is the YOLO segmentation format: images/{train,val}/*.jpg, labels/{train,val}/*.txt (one polygon per line).
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--count", type=int, default=400)
p.add_argument("--out", default="datasets/pinky_road")
p.add_argument("--num_envs", type=int, default=16, help="cities photographed at the same time")
p.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.perception = "yolo"                       # full city + the robot camera
app = AppLauncher(args).app

import math  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf  # noqa: E402

from rl_shortcut import ROOT, pink_city, pinky_env, shutdown  # noqa: E402
from rl_shortcut.road_seg import CLASSES, mask_to_view  # noqa: E402

COLORS = np.array([(40, 40, 48), (90, 96, 110), (232, 150, 190)], dtype=np.uint8)   # background, road, sidewalk


def sample_poses(n, road_points, gen, device):
    """70 %: on the ring road, up to 17 cm off its centre line, heading along it +-35 deg (either way round).
    30 %: anywhere on a road, any heading."""
    angle = torch.rand(n, generator=gen) * 2 * math.pi
    xy, heading = pink_city.ring_point(angle, (torch.rand(n, generator=gen) * 2 - 1) * 0.17)
    yaw = heading + (torch.rand(n, generator=gen) * 2 - 1) * 0.6 + math.pi * (torch.rand(n, generator=gen) < 0.3)
    anywhere = torch.rand(n, generator=gen) < 0.3
    pick = road_points[torch.randint(len(road_points), (n,), generator=gen)]
    xy = torch.where(anywhere[:, None], pick, xy)
    yaw = torch.where(anywhere, torch.rand(n, generator=gen) * 2 * math.pi, yaw)
    return xy.to(device), yaw.to(device)


def label_masks(camera, origins, labels):
    """Class of every pixel: 0 background, 1 road, 2 sidewalk."""
    depth = camera.data.output["distance_to_image_plane"][..., 0]
    n, h, w = depth.shape
    k = camera.data.intrinsic_matrices
    v, u = torch.meshgrid(torch.arange(h, device=depth.device) + 0.5, torch.arange(w, device=depth.device) + 0.5,
                          indexing="ij")
    # the point a pixel shows, in the camera frame (x right, y down, z ahead) and then in its city
    cam = torch.stack([(u - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None] * depth,
                       (v - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None] * depth, depth], dim=-1).view(n, -1, 3)
    points = quat_apply(camera.data.quat_w_ros[:, None, :].expand(-1, h * w, -1), cam) + camera.data.pos_w[:, None, :]
    points = points - origins[:, None, :]
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    size = labels.shape[0]
    col = ((x + pink_city.SIZE / 2) / pink_city.SIZE * size).long().clamp(0, size - 1)
    row = ((pink_city.SIZE / 2 - y) / pink_city.SIZE * size).long().clamp(0, size - 1)
    floor = torch.isfinite(z) & (z.abs() < 0.012) & (x.abs() < pink_city.SIZE / 2) & (y.abs() < pink_city.SIZE / 2)
    kind = labels[row, col]
    out = torch.where(kind >= 2, 1, torch.where(kind == 1, 2, 0)) * floor
    return out.view(n, h, w).to(torch.uint8)


def polygons(mask):
    """YOLO label lines of one image: 'class x1 y1 x2 y2 ...' with coordinates from 0 to 1."""
    h, w = mask.shape
    lines = []
    for cls in range(len(CLASSES)):
        contours, _ = cv2.findContours((mask == cls + 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 80:
                continue
            c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
            if len(c) >= 3:
                lines.append(f"{cls} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c))
    return lines


def main():
    cfg = pinky_env.make_env_cfg(args)
    cfg.scene.num_envs = args.num_envs
    cfg.scene.camera.data_types = ["rgb", "distance_to_image_plane"]
    cfg.scene.camera.update_latest_camera_pose = True
    cfg.perception = "map"                     # no YOLO yet: this script makes its training data
    cfg.seed = args.seed
    env = ManagerBasedEnv(cfg=cfg)
    robot, camera = env.scene["robot"], env.scene["camera"]
    origins = env.scene.env_origins
    road_map = pinky_env.road_map(env)
    labels = torch.from_numpy(np.array(Image.open(pink_city.LABELS_PNG))).to(env.device)
    rows, cols = torch.nonzero(labels.cpu() >= 2, as_tuple=True)
    road_points = torch.stack([(cols + 0.5) / labels.shape[0] * pink_city.SIZE - pink_city.SIZE / 2,
                               pink_city.SIZE / 2 - (rows + 0.5) / labels.shape[0] * pink_city.SIZE], dim=-1)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sun = env.sim.stage.GetPrimAtPath("/World/sun")

    out = ROOT / args.out
    for part in ("images/train", "images/val", "labels/train", "labels/val", "preview"):
        (out / part).mkdir(parents=True, exist_ok=True)
    print(f"[dataset] {args.count} images -> {out}")
    saved, share, agree = 0, np.zeros(3), []
    zero = torch.zeros(env.num_envs, 2, device=env.device)
    while saved < args.count:
        xy, yaw = sample_poses(env.num_envs, road_points, gen, env.device)
        pose = robot.data.default_root_state[:, :7].clone()
        pose[:, :2] = origins[:, :2] + xy
        pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
        # a different sun for every round: direction and brightness
        sun.GetAttribute("xformOp:orient").Set(Gf.Quatd(Gf.Rotation(Gf.Vec3d(0, 0, 1), rng.uniform(0, 360)).GetQuat())
                                               * Gf.Quatd(Gf.Rotation(Gf.Vec3d(1, 0, 0), rng.uniform(15, 60)).GetQuat()))
        sun.GetAttribute("inputs:intensity").Set(float(rng.uniform(600.0, 2500.0)))
        for _ in range(4):                     # let the robot settle and the picture sharpen
            env.step(zero)
        rgb = camera.data.output["rgb"][..., :3]
        masks = label_masks(camera, origins, labels)
        # check: the road view computed from the map (training) against the one from this camera image (play)
        seen = mask_to_view(masks == 1)
        xy_now = robot.data.root_pos_w[:, :2] - origins[:, :2]
        agree.append((road_map.view(xy_now, robot.data.heading_w) - seen).abs().mean().item())
        for i in range(env.num_envs):
            if saved >= args.count:
                break
            image, mask = rgb[i].cpu().numpy(), masks[i].cpu().numpy()
            part = "val" if saved % 10 == 9 else "train"
            cv2.imwrite(str(out / f"images/{part}/{saved:05d}.jpg"), image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / f"labels/{part}/{saved:05d}.txt").write_text("\n".join(polygons(mask)) + "\n")
            if saved < 16:
                color = COLORS[mask]
                blend = (image * 0.45 + color * 0.55).astype(np.uint8)
                cv2.imwrite(str(out / f"preview/{saved:05d}.jpg"), np.concatenate([image, color, blend], axis=1)[:, :, ::-1])
            share += np.bincount(mask.ravel(), minlength=3)
            saved += 1
        print(f"[dataset] {saved} / {args.count}", flush=True)
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
                                   + "".join(f"  {k}: {name}\n" for k, name in enumerate(CLASSES)))
    share = share / share.sum() * 100
    print(f"[dataset] pixels: background {share[0]:.1f} %, road {share[1]:.1f} %, sidewalk {share[2]:.1f} %")
    print(f"[dataset] road view from the map vs from the camera labels: mean difference {np.mean(agree):.3f} (0 = same)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    shutdown(app)
