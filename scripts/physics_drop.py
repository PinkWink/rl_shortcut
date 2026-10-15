"""Drop three balls (restitution 0.0, 0.5, 0.9) from 1 m and measure fall time and bounces.

  python scripts/physics_drop.py                     # watch in the viewer
  python scripts/physics_drop.py --headless --dt 0.02
Theory: fall time sqrt(2h/g) = 0.452 s, bounce height = e^2 * h.
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--dt", type=float, default=0.005, help="physics step in seconds")
p.add_argument("--drop_height", type=float, default=1.0)
p.add_argument("--seconds", type=float, default=3.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from rl_shortcut import shutdown  # noqa: E402

RADIUS = 0.1
RESTITUTION = [0.0, 0.5, 0.9]
COLORS = [(0.8, 0.2, 0.2), (0.9, 0.7, 0.1), (0.2, 0.6, 0.9)]


def ball(i, e):
    return RigidObjectCfg(
        prim_path=f"/World/ball_{i}",
        spawn=sim_utils.SphereCfg(
            radius=RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # "max": the ball's restitution wins over the ground's 0
            physics_material=sim_utils.RigidBodyMaterialCfg(restitution=e, restitution_combine_mode="max"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[i]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5 * (i - 1), args.drop_height + RADIUS)),
    )


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt))
    sim.set_camera_view(eye=(3.0, 0.0, 1.0), target=(0.0, 0.0, 0.5))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    balls = [RigidObject(ball(i, e)) for i, e in enumerate(RESTITUTION)]
    sim.reset()

    hit = [None] * 3                  # time of first ground contact
    peak = [0.0] * 3                  # highest point after the first bounce
    prev_vz = [0.0] * 3
    for step in range(int(args.seconds / args.dt)):
        sim.step()
        t = (step + 1) * args.dt
        for i, b in enumerate(balls):
            b.update(args.dt)
            z = float(b.data.root_pos_w[0, 2]) - RADIUS
            vz = float(b.data.root_lin_vel_w[0, 2])
            if hit[i] is None and vz - prev_vz[i] > 1.0:   # sudden upward change = hit the ground
                hit[i] = t
            elif hit[i] is not None and t < hit[i] + 1.0:
                peak[i] = max(peak[i], z)
            prev_vz[i] = vz
        if not app.is_running():
            return

    print(f"dt = {args.dt} s, drop height {args.drop_height} m, theory: fall {math.sqrt(2 * args.drop_height / 9.81):.3f} s")
    for i, e in enumerate(RESTITUTION):
        print(f"restitution {e:.1f}: fall {hit[i] or 0:.3f} s, bounce {peak[i]:.3f} m (theory {e * e * args.drop_height:.3f} m)")
    while app.is_running() and not args.headless:
        sim.step()


if __name__ == "__main__":
    main()
    shutdown(app)
