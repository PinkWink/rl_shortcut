"""Train a policy with PPO (RSL-RL).

  python scripts/train.py --task pendulum --headless
  python scripts/train.py --task go2 --headless
  python scripts/train.py --task g1 --motion walk --headless
  python scripts/train.py --task g1 --motion bruce_lee --headless --max_iterations 1000

Checkpoints: logs/<experiment>/<date_time>/model_<iteration>.pt (experiment = pendulum, go2, g1_walk, ...)
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["pendulum", "go2", "g1"])
p.add_argument("--motion", default="walk", help="g1 only: motions/<motion>.npz")
p.add_argument("--num_envs", type=int, default=None)
p.add_argument("--max_iterations", type=int, default=None)
p.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

# Isaac Lab modules can be imported only after the app has started
import importlib  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from rl_shortcut import LOG_DIR, shutdown  # noqa: E402


def main():
    task = importlib.import_module(f"rl_shortcut.{args.task}_env")
    env_cfg = task.make_env_cfg(args)
    runner_cfg = task.make_runner_cfg(args)
    if args.num_envs:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations:
        runner_cfg.max_iterations = args.max_iterations
    env_cfg.seed = runner_cfg.seed = args.seed

    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"[train] {runner_cfg.experiment_name}: {env.num_envs} envs, "
          f"observation {env.observation_manager.group_obs_dim['policy'][0]}, action {env.action_manager.total_action_dim}")
    log_dir = LOG_DIR / runner_cfg.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env), runner_cfg.to_dict(), log_dir=str(log_dir), device="cuda:0")

    t0 = time.time()
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min -> {log_dir}")
    env.close()


if __name__ == "__main__":
    main()
    shutdown(app)
