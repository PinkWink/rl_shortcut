"""Run a trained policy and print how well it does.

  python scripts/play.py --task pendulum
  python scripts/play.py --task go2 --cmd 1.0 0.0 0.0           # walk forward at 1 m/s
  python scripts/play.py --task g1 --motion walk
  python scripts/play.py --task g1 --motion bruce_lee --checkpoint g1_walk   # walking policy, kung fu clip
  python scripts/play.py --task go2 --headless --enable_cameras --video outputs/go2.mp4 --seconds 10

--checkpoint: a .pt path or an experiment name (pendulum, go2, g1_walk, g1_bruce_lee).
By name it takes the newest logs/<name>/*/model_*.pt, else the shipped checkpoints/<name>.pt.
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["pendulum", "go2", "g1"])
p.add_argument("--motion", default="walk", help="g1 only: motions/<motion>.npz")
p.add_argument("--checkpoint", default=None)
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"), help="go2 only")
p.add_argument("--seconds", type=float, default=None, help="stop after this many simulated seconds")
p.add_argument("--video", default=None, help="save env 0 as mp4 (needs --enable_cameras)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import importlib  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from rl_shortcut.ppo import latest_checkpoint  # noqa: E402
from rl_shortcut import shutdown  # noqa: E402


def main():
    task = importlib.import_module(f"rl_shortcut.{args.task}_env")
    env_cfg = task.make_env_cfg(args)
    runner_cfg = task.make_runner_cfg(args)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.observations.policy.enable_corruption = False
    if args.cmd is not None:          # go2: a fixed command instead of random ones
        vx, vy, wz = args.cmd
        c = env_cfg.commands.base_velocity
        c.rel_standing_envs = 0.0
        c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = (vx, vx), (vy, vy), (wz, wz)
    ckpt = Path(args.checkpoint) if args.checkpoint and args.checkpoint.endswith(".pt") \
        else latest_checkpoint(args.checkpoint or runner_cfg.experiment_name)
    if ckpt is None:
        raise SystemExit(f"no checkpoint for {args.checkpoint or runner_cfg.experiment_name}: train first")

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(str(ckpt), load_optimizer=False)
    policy = runner.get_inference_policy(device="cuda:0")
    print(f"[play] {ckpt}")

    report = Report(env, args)
    frames = []
    obs = wrapped.get_observations()
    steps = int(args.seconds / env.step_dt) if args.seconds else None
    step = 0
    while app.is_running() and (steps is None or step < steps):
        with torch.inference_mode():
            obs, _, dones, extras = wrapped.step(policy(obs))
        report.update(dones, extras)
        if args.video:
            frames.append(env.render())
        step += 1
    report.final()
    if args.video:
        import imageio
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=round(1 / env.step_dt))
        print(f"[play] saved {args.video} ({len(frames)} frames)")
    env.close()


class Report:
    """Once per simulated second: early episode ends (falls) so far, and how well the task goes."""

    def __init__(self, env, args):
        self.env, self.args = env, args
        self.steps_per_s = round(1 / env.step_dt)
        self.step, self.falls, self.values = 0, 0, []

    def update(self, dones, extras):
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
        self.falls += int((dones.bool() & ~timeouts).sum())
        self.values.append(self.measure())
        self.step += 1
        if self.step % self.steps_per_s == 0:
            recent = torch.stack(self.values[-self.steps_per_s:]).mean(0)
            print(f"[play] t={self.step // self.steps_per_s:3d} s  fails={self.falls:3d}  {self.text(recent)}")

    def measure(self):
        robot = self.env.scene["robot"]
        if self.args.task == "go2":       # measured forward, sideways and turning speed (mean of envs)
            v = robot.data.root_lin_vel_b[:, :2].mean(0)
            return torch.stack([v[0], v[1], robot.data.root_ang_vel_b[:, 2].mean()])
        if self.args.task == "g1":        # RMS joint angle error to the clip
            from rl_shortcut.g1_env import ref
            r = ref(self.env)
            return (robot.data.joint_pos[:, r.ids] - r.motion.dof[r.frame(self.env)]).square().mean().sqrt()
        return (robot.data.joint_pos[:, 1] - torch.pi).abs().mean()   # pendulum: |angle from upright|

    def text(self, x):
        if self.args.task == "go2":
            return f"vx {x[0]:+.2f} m/s  vy {x[1]:+.2f} m/s  wz {x[2]:+.2f} rad/s"
        if self.args.task == "g1":
            return f"joint error {float(x):.3f} rad"
        return f"pendulum angle from upright {float(x):.3f} rad"

    def final(self):
        print(f"[play] {self.step / self.steps_per_s:.0f} s x {self.env.num_envs} envs: fails={self.falls}, "
              f"{self.text(torch.stack(self.values).mean(0))}")


if __name__ == "__main__":
    main()
    shutdown(app)
