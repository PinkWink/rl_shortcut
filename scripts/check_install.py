"""Check the install: versions, GPU, and 100 physics steps of an empty scene.

  python scripts/check_install.py --headless
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import importlib.metadata as md  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402

from rl_shortcut import shutdown  # noqa: E402

for pkg in ("isaacsim", "isaaclab", "rsl-rl-lib", "torch"):
    print(f"{pkg:12s} {md.version(pkg)}")
print(f"{'GPU':12s} {torch.cuda.get_device_name(0)}, cuda {torch.cuda.is_available()}")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim.reset()
for _ in range(100):
    sim.step()
print(f"physics: 100 steps of {sim.get_physics_dt()} s on {sim.device}")
shutdown(app)
