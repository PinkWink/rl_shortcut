"""Many Go2 robots in one GPU physics scene: how fast can PhysX step them?

  python scripts/physics_parallel.py --num_envs 16                 # watch
  python scripts/physics_parallel.py --headless --num_envs 4096
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import time  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from rl_shortcut.go2_env import GO2_CFG  # noqa: E402
from rl_shortcut import shutdown  # noqa: E402


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")   # copied into env_0, env_1, ...


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=(6.0, 6.0, 4.0), target=(0.0, 0.0, 0.0))
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=1.5))
    sim.reset()
    robot = scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins                      # each robot at its own grid cell
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    target = robot.data.default_joint_pos.clone()        # hold the standing pose

    for i in range(args.steps + 50):
        if i == 50:                                       # skip warm-up steps
            torch.cuda.synchronize()
            t0 = time.time()
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step(render=not args.headless)
        scene.update(sim.get_physics_dt())
    torch.cuda.synchronize()
    dt = time.time() - t0
    z = robot.data.root_pos_w[:, 2]
    print(f"{args.num_envs} robots: {args.steps / dt:.0f} steps/s, {args.num_envs * args.steps / dt:,.0f} robot-steps/s, "
          f"{args.steps * 0.005 / dt:.1f}x real time; body height {z.mean():.3f} m")
    while app.is_running() and not args.headless:
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    shutdown(app)
