"""Convert the Pinky Pro URDF to the USD the course uses.

  python tools/convert_pinky.py --description ~/pinky_pro/src/pinky_pro/pinky_description
      -> assets/pinky_pro/usd/pinky.usd

The URDF comes from the official repository https://github.com/pinklab-art/pinky_pro (pinky_description;
`xacro urdf/robot.urdf.xacro > urdf/pinky.urdf`). The course repository ships only the converted USD.
Fixed joints are merged, so the robot is one body + two drive wheels + the rear caster (swivel and wheel).
"""
import argparse
import re
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--description", required=True, help="pinky_description folder (urdf/pinky.urdf, meshes/)")
p.add_argument("--out", default="assets/pinky_pro/usd")
args = p.parse_args()
app = AppLauncher(headless=True).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from rl_shortcut import ROOT, shutdown  # noqa: E402

description = Path(args.description).expanduser().resolve()
urdf = (description / "urdf/pinky.urdf").read_text()
urdf = re.sub(r"package://pinky_description/", f"{description}/", urdf)      # mesh paths the importer can open
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "pinky.urdf"
    path.write_text(urdf)
    cfg = UrdfConverterCfg(
        asset_path=str(path),
        usd_dir=str(ROOT / args.out),
        usd_file_name="pinky.usd",
        fix_base=False,                      # a mobile robot
        merge_fixed_joints=True,             # sensor mounts become part of the body
        collider_type="convex_hull",
        self_collision=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(      # gains are set again by the actuators in pinky_env.py
            drive_type="force", target_type="velocity",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.05)),
    )
    print("[convert]", UrdfConverter(cfg).usd_path)
shutdown(app)
