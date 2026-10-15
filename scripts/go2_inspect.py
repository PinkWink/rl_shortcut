"""Open go2.usd and list what is inside: links (mass), joints (limits), then the joint order Isaac Lab uses.

  python scripts/go2_inspect.py --headless
  python scripts/go2_inspect.py              # also opens the viewer (Stage tab shows the same tree)
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

from rl_shortcut.go2_env import GO2_CFG, GO2_USD  # noqa: E402
from rl_shortcut import shutdown  # noqa: E402


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=(1.2, 1.2, 0.8), target=(0.0, 0.0, 0.3))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    robot = Articulation(GO2_CFG.replace(prim_path="/World/Go2"))
    print(f"USD: {GO2_USD}")

    stage = sim_utils.get_current_stage()
    links, joints = [], []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/Go2"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append((prim.GetName(), UsdPhysics.MassAPI(prim).GetMassAttr().Get() or 0.0))
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            joints.append((prim.GetName(), j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()))

    print(f"\n{len(links)} links, total {sum(m for _, m in links):.2f} kg")
    for name, mass in links:
        print(f"  {name:16s} {mass:6.3f} kg")
    print(f"\n{len(joints)} revolute joints (limits in degrees, from the USD)")
    for name, lo, hi in joints:
        print(f"  {name:16s} {lo:7.1f} .. {hi:6.1f}")

    sim.reset()
    print("\nIsaac Lab joint order (actions and observations use this order) and the standing pose:")
    for i, (name, q) in enumerate(zip(robot.data.joint_names, robot.data.default_joint_pos[0].tolist())):
        print(f"  {i:2d} {name:16s} {q:+.2f} rad")
    while app.is_running() and not args.headless:
        sim.step()


if __name__ == "__main__":
    main()
    shutdown(app)
