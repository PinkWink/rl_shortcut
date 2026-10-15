"""Drop the Go2 and let PD motors hold the standing pose. Try different stiffness (kp).

  python scripts/go2_stand.py                    # kp 25 (the value used for training)
  python scripts/go2_stand.py --kp 5             # soft motors: the body sags
  python scripts/go2_stand.py --kp 0             # no motor: it collapses
torque = kp * (target - q) - kd * dq, limited to 23.5 Nm
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--kp", type=float, default=25.0)
p.add_argument("--kd", type=float, default=0.5)
p.add_argument("--seconds", type=float, default=3.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from rl_shortcut.go2_env import GO2_CFG  # noqa: E402
from rl_shortcut import shutdown  # noqa: E402


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=(1.5, 1.5, 0.8), target=(0.0, 0.0, 0.25))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    cfg = GO2_CFG.replace(prim_path="/World/Go2")
    cfg.actuators["legs"].stiffness = args.kp
    cfg.actuators["legs"].damping = args.kd
    robot = Articulation(cfg)
    sim.reset()
    # start in the standing pose (the USD itself has straight legs, which would poke into the ground)
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    target = robot.data.default_joint_pos.clone()

    print(f"kp {args.kp}, kd {args.kd}")
    steps = int(args.seconds / 0.005)
    for i in range(steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(0.005)
        if (i + 1) % 100 == 0:
            z = float(robot.data.root_pos_w[0, 2])
            err = (robot.data.joint_pos - target).abs().max().item()
            tau = robot.data.applied_torque.abs().max().item()
            print(f"t={(i + 1) * 0.005:.1f} s  body height {z:.3f} m  max joint error {err:.3f} rad  max torque {tau:5.1f} Nm")
    while app.is_running() and not args.headless:
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(0.005)


if __name__ == "__main__":
    main()
    shutdown(app)
