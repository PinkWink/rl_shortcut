"""Play a reference clip on the G1 without physics control: this is the target the policy imitates.

  python scripts/g1_motion.py --motion walk
  python scripts/g1_motion.py --motion bruce_lee
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--motion", default="walk", help="motions/<motion>.npz")
p.add_argument("--loops", type=int, default=0, help="stop after this many loops (0: until the window closes)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import time  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # noqa: E402

from rl_shortcut.motion import Motion  # noqa: E402
from rl_shortcut import shutdown  # noqa: E402


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, render_interval=4))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    robot = Articulation(G1_MINIMAL_CFG.replace(prim_path="/World/G1"))
    sim.reset()
    m = Motion(args.motion, sim.device)
    ids = m.joint_ids(robot.data.joint_names)
    print(f"{args.motion}: {m.T} frames @ {m.fps:.0f} fps = {m.duration:.2f} s, "
          f"pelvis height {m.root_pos[:, 2].min():.2f}..{m.root_pos[:, 2].max():.2f} m, "
          f"travel {(m.root_pos[-1, :2] - m.root_pos[0, :2]).norm():.2f} m")

    step, policy_dt = 0, 0.02                      # show the clip at 50 Hz like the policy
    t0 = time.time()
    while app.is_running():
        t = step * policy_dt
        f = int(t * m.fps) % m.T
        if args.loops and t >= args.loops * m.duration:
            break
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[:, ids] = m.dof[f]
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        robot.write_root_pose_to_sim(torch.cat([m.root_pos[f], m.root_quat[f]]).unsqueeze(0))
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim.device))
        root = m.root_pos[f].tolist()
        sim.set_camera_view(eye=(root[0] + 2.5, root[1] + 2.5, 1.5), target=(root[0], root[1], 0.7))
        for _ in range(4):
            sim.step(render=False)
        sim.render()
        robot.update(policy_dt)
        step += 1
        if not args.headless:                      # play in real time
            time.sleep(max(0.0, t0 + step * policy_dt - time.time()))


if __name__ == "__main__":
    main()
    shutdown(app)
