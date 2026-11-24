"""Screenshots for the Pinky Pro pages, rendered headless from the same code the students run.

  .venv/bin/python docs/lecture/render_pinky.py city  --headless --enable_cameras
  .venv/bin/python docs/lecture/render_pinky.py drive --headless --enable_cameras
  .venv/bin/python docs/lecture/render_pinky.py train --headless --enable_cameras
Raw frames go to docs/lecture/raw/pinky_*.png; draw_lesson11_12_figures.py lays them out.
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("shot", choices=["city", "drive", "train"])
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.perception = "map" if args.shot == "train" else "yolo"
app = AppLauncher(args).app

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from rl_shortcut import pinky_env, shutdown  # noqa: E402
from rl_shortcut.ppo import latest_checkpoint  # noqa: E402
from rl_shortcut.road_seg import debug_panel  # noqa: E402

RAW = Path(__file__).resolve().parent / "raw"
RAW.mkdir(exist_ok=True)


def save(image, name):
    iio.imwrite(RAW / f"{name}.png", np.asarray(image)[..., :3])
    print("saved", name, flush=True)


def make(num_envs):
    cfg = pinky_env.make_env_cfg(args)
    cfg.scene.num_envs = num_envs
    cfg.observations.policy.enable_corruption = False
    env = ManagerBasedRLEnv(cfg=cfg, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, pinky_env.make_runner_cfg(args).to_dict(), log_dir=None, device="cuda:0")
    runner.load(str(latest_checkpoint("pinky")), load_optimizer=False)
    return env, wrapped, runner.get_inference_policy(device="cuda:0")


def place(env, x, y, yaw):
    robot = env.scene["robot"]
    pose = robot.data.default_root_state[:, :7].clone()
    pose[:, 0], pose[:, 1] = env.scene.env_origins[:, 0] + x, env.scene.env_origins[:, 1] + y
    yaw = torch.full((env.num_envs,), yaw, device=env.device)
    pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))


def look(env, eye, target, steps=6):
    env.sim.set_camera_view(eye, target)
    zero = torch.zeros(env.num_envs, 2, device=env.device)
    for _ in range(steps):
        env.step(zero)
    for _ in range(3):
        env.render()
    return env.render()


def city():
    env, wrapped, policy = make(1)
    wrapped.get_observations()
    spots = {"a": (-1.15, 0.2, -math.pi / 2), "b": (-0.95, -1.06, -0.5), "c": (0.3, -1.15, 0.0)}
    place(env, *spots["a"])
    for _ in range(80):                    # textures load in the background: wait for them
        env.render()
    save(look(env, (-2.0, -2.7, 2.4), (0.0, 0.05, 0.0), 12), "pinky_city_overview")
    save(look(env, (0.0, -0.01, 4.4), (0.0, 0.0, 0.0)), "pinky_city_top")
    save(look(env, (-1.42, 0.50, 0.22), (-1.15, 0.2, 0.05)), "pinky_robot")
    save(look(env, (-1.15, 0.75, 0.22), (-1.15, -0.4, 0.03)), "pinky_street")
    views = {}
    for name, spot in spots.items():
        place(env, *spot)
        look(env, (-2.0, -2.7, 2.4), (0.0, 0.05, 0.0))
        last = env._road_seg.last
        xy, yaw = pinky_env.city_pose(env)
        save(last["rgb"][0].cpu().numpy(), f"pinky_camera_{name}")
        views[f"mask_{name}"] = last["mask"][0].cpu().numpy()
        views[f"yolo_{name}"] = last["view"][0].view(8, 16).cpu().numpy()
        views[f"map_{name}"] = pinky_env.road_map(env).view(xy, yaw)[0].view(8, 16).cpu().numpy()
    np.savez(RAW / "pinky_views.npz", **views)
    env.close()


def drive():
    env, wrapped, policy = make(1)
    env.sim.set_camera_view((-2.0, -2.7, 2.4), (0.0, 0.05, 0.0))
    for _ in range(80):                    # textures load in the background: wait for them
        env.render()
    obs = wrapped.get_observations()
    grab = {int(t / env.step_dt): k for k, t in enumerate((2, 8, 14, 20, 26, 32))}
    track = []
    for step in range(int(36 / env.step_dt) + 1):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs))
        xy, _ = pinky_env.city_pose(env)
        track.append(xy[0].cpu().numpy().copy())
        if step in grab:
            env.render()
            save(debug_panel(env, env.render()), f"pinky_yolo_{grab[step]}")
    np.save(RAW / "pinky_track.npy", np.array(track))
    env.close()


def train():
    env, wrapped, policy = make(16)
    env.sim.set_camera_view((-4.2, -11.5, 6.5), (-0.3, -1.8, 0.0))
    obs = wrapped.get_observations()
    for _ in range(int(3 / env.step_dt)):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs))
    for _ in range(80):                    # textures load in the background: wait for them
        env.render()
    save(env.render(), "pinky_train")
    env.close()


if __name__ == "__main__":
    {"city": city, "drive": drive, "train": train}[args.shot]()
    shutdown(app)
