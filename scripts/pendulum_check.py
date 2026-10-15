"""Check the kit model: joints, masses, and the free swing period of the pendulum.

  python scripts/pendulum_check.py               # arm held still, watch the pendulum swing
  python scripts/pendulum_check.py --free_arm    # no motor torque: the arm swings back and forth too
Theory (small swings, arm held): T = 2 pi sqrt(I / (m g d)), I about the pivot, d = pivot to center of mass.
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--start_deg", type=float, default=30.0, help="pendulum angle from hanging down")
p.add_argument("--seconds", type=float, default=5.0)
p.add_argument("--free_arm", action="store_true", help="motor off instead of holding the arm at 0")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from rl_shortcut import shutdown  # noqa: E402
from rl_shortcut.pendulum_env import PENDULUM_CFG  # noqa: E402

# pendulum link in the URDF: mass, center of mass from the pivot (y, z), inertia about the swing axis (y)
M, COM_Z, IYY = 0.016, -0.06737, 2.283e-05


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=(0.45, 0.45, 0.3), target=(0.0, 0.0, 0.12))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    cfg = PENDULUM_CFG.replace(prim_path="/World/Pendulum")
    if not args.free_arm:                       # hold the arm with a stiff position controller
        cfg.actuators["arm_motor"].stiffness, cfg.actuators["arm_motor"].damping = 1e4, 10.0
        cfg.actuators["arm_motor"].effort_limit_sim = 100.0
    robot = Articulation(cfg)
    sim.reset()

    print("joints:", robot.data.joint_names)
    print("links:", ", ".join(f"{n} {m * 1000:.1f} g" for n, m in zip(robot.data.body_names, robot.data.default_mass[0])))
    print("joint limits (rad):", [tuple(round(v, 2) for v in lim) for lim in robot.data.joint_pos_limits[0].tolist()])

    q = torch.tensor([[0.0, math.radians(args.start_deg)]], device=sim.device)   # arm, pendulum (0 = hanging)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    crossings, prev, t, arm_max = [], None, 0.0, 0.0
    for _ in range(int(args.seconds / 0.005)):
        robot.set_joint_position_target(torch.zeros_like(q))   # arm target 0 (ignored with --free_arm)
        robot.write_data_to_sim()
        sim.step()
        robot.update(0.005)
        t += 0.005
        theta = float(robot.data.joint_pos[0, 1])
        arm_max = max(arm_max, abs(float(robot.data.joint_pos[0, 0])))
        if prev is not None and prev > 0 >= theta:             # passes the bottom going one way
            crossings.append(t)
        prev = theta
    period = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
    inertia = IYY + M * COM_Z ** 2
    theory = 2 * math.pi * math.sqrt(inertia / (M * 9.81 * abs(COM_Z)))
    print(f"{'arm free' if args.free_arm else 'arm held'}, swing from {args.start_deg:.0f} deg: "
          f"period {period:.3f} s (theory with the arm held {theory:.3f} s), arm moved up to {math.degrees(arm_max):.1f} deg")
    while app.is_running() and not args.headless:
        sim.step()


if __name__ == "__main__":
    main()
    shutdown(app)
