"""Record the clips of the course introduction video: 1920 x 1080, straight from the course code and checkpoints.

  .venv/bin/python -u docs/intro/record_clips.py <shot> --headless --enable_cameras
shots: pendulum_kit, pendulum_many, pinky_train, pinky_learn, pinky_yolo, go2_play, go2_forward, g1_walk, g1_bruce_lee, g1_walk_on_bruce_lee
Clips go to outputs/intro/<shot>.mp4 (pinky_yolo also writes pinky_city.mp4: the viewport alone).
Learning progress: add --checkpoint logs/<experiment>/<run>/model_<iteration>.pt --name <shot>_it<iteration> --seconds 7
"""
import argparse

from isaaclab.app import AppLauncher

#        task        motion        checkpoint  envs  seconds  camera (eye, lookat; None = follow the robot)
SHOTS = {
    "pendulum_kit": ("pendulum", "walk", "pendulum", 1, 10.0, ((0.24, 0.46, 0.24), (0.0, 0.02, 0.10))),
    "pendulum_many": ("pendulum", "walk", "pendulum", 16, 8.0, ((0.8, 2.0, 0.9), (0.0, 0.0, -0.05))),
    "pinky_train": ("pinky", "walk", "pinky", 16, 14.0, ((-4.2, -11.5, 6.5), (-0.3, -1.8, 0.0))),
    "pinky_learn": ("pinky", "walk", "pinky", 4, 12.0, ((0.0, -5.6, 5.6), (0.0, -0.55, 0.0))),
    "pinky_yolo": ("pinky", "walk", "pinky", 1, 38.0, ((-2.0, -2.7, 2.4), (0.0, 0.05, 0.0))),
    "go2_play": ("go2", "walk", "go2", 16, 14.0, ((7.5, 7.5, 5.0), (0.0, 0.0, 0.0))),
    "go2_forward": ("go2", "walk", "go2", 1, 9.0, None),
    "g1_walk": ("g1", "walk", "g1_walk", 1, 9.0, None),
    "g1_bruce_lee": ("g1", "bruce_lee", "g1_bruce_lee", 1, 11.0, None),
    "g1_walk_on_bruce_lee": ("g1", "bruce_lee", "g1_walk", 1, 5.0, None),
}

p = argparse.ArgumentParser()
p.add_argument("shot", choices=list(SHOTS))
p.add_argument("--checkpoint", default=None, help="a .pt path instead of the newest one (learning-progress clips)")
p.add_argument("--name", default=None, help="output name (default: the shot name)")
p.add_argument("--seconds", type=float, default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.motion = SHOTS[args.shot][1]
args.perception = "yolo" if args.shot == "pinky_yolo" else "map"
app = AppLauncher(args).app

import importlib  # noqa: E402

import imageio  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from rl_shortcut import ROOT, shutdown  # noqa: E402
from rl_shortcut.ppo import latest_checkpoint  # noqa: E402

OUT = ROOT / "outputs/intro"


def main():
    task, _, checkpoint, num_envs, seconds, camera = SHOTS[args.shot]
    mod = importlib.import_module(f"rl_shortcut.{task}_env")
    env_cfg, runner_cfg = mod.make_env_cfg(args), mod.make_runner_cfg(args)
    env_cfg.scene.num_envs = num_envs
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.viewer.resolution = (1920, 1080)
    if args.shot == "go2_forward":
        c = env_cfg.commands.base_velocity
        c.rel_standing_envs, c.debug_vis = 0.0, False
        c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = (1.0, 1.0), (0.0, 0.0), (0.0, 0.0)
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (0.6, -2.2, 0.7), (0.3, 0.0, 0.2)
    if task == "g1":
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (0.8, -3.2, 0.0), (0.0, 0.0, -0.15)      # offsets from the pelvis
    if camera:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye, env_cfg.viewer.lookat = camera
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(str(args.checkpoint or latest_checkpoint(checkpoint)), load_optimizer=False)
    name, seconds = args.name or args.shot, args.seconds or seconds
    policy = runner.get_inference_policy(device="cuda:0")

    for _ in range(90):                        # textures load in the background: the first frames are black
        env.render()
    every = max(1, round(1 / env.step_dt / 50))        # at most 50 frames per second
    fps = round(1 / env.step_dt / every)
    OUT.mkdir(parents=True, exist_ok=True)
    writers = {"main": imageio.get_writer(OUT / f"{name}.mp4", fps=fps, quality=9, macro_block_size=1)}
    if args.shot == "pinky_yolo":
        from rl_shortcut.road_seg import debug_panel
        writers["city"] = imageio.get_writer(OUT / "pinky_city.mp4", fps=fps, quality=9, macro_block_size=1)
    obs = wrapped.get_observations()
    for step in range(int(seconds / env.step_dt)):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs))
        if step % every == 0:
            image = env.render()
            if "city" in writers:
                writers["city"].append_data(image)
                image = debug_panel(env, image)
            writers["main"].append_data(image)
    for w in writers.values():
        w.close()
    print(f"[record] {name}: {seconds:g} s at {fps} fps -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    shutdown(app)
