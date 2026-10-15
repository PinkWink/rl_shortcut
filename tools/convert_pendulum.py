"""Convert the PinkLAB pendulum kit URDF to the USD the course uses.

  python tools/convert_pendulum.py            # -> assets/pinklab_pendulum/usd/model_v3.usd

The kit URDF has visual meshes only. Colliders are made from the visual meshes (convex hulls):
with no collision shapes on its links, PhysX takes about 4 minutes to start 4096 pendulums instead of seconds.
Nothing touches them (self-collision off, the base is fixed), so they do not change the motion.
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--out", default="assets/pinklab_pendulum/usd")
args = p.parse_args()
app = AppLauncher(headless=True).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from rl_shortcut import ROOT, shutdown  # noqa: E402

cfg = UrdfConverterCfg(
    asset_path=str(ROOT / "assets/pinklab_pendulum/model_v3.urdf"),
    usd_dir=str(ROOT / args.out),
    usd_file_name="model_v3.usd",
    fix_base=True,                       # the base sits on the desk
    collision_from_visuals=True,         # the URDF has no <collision>; use the visual meshes
    collider_type="convex_hull",
    self_collision=False,
    force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force", target_type="none",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
)
print("[convert]", UrdfConverter(cfg).usd_path)
shutdown(app)
