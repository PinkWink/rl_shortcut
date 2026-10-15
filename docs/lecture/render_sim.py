"""Screenshots for the lecture pages, rendered headless from the same code the students run.

  .venv/bin/python docs/lecture/render_sim.py <shot> --headless --enable_cameras
shots: drop, parallel, stand, go2, cartpole, go2_play, go2_forward, g1, g1_ref_walk, g1_ref_bruce_lee,
       g1_play_walk, g1_walk_on_bruce_lee, g1_play_bruce_lee
Raw frames go to docs/lecture/raw/<shot>_*.png; draw_renders.py lays them out into l0x_*.png.
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("shot")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402

from rl_shortcut import shutdown  # noqa: E402

RAW = Path(__file__).resolve().parent / "raw"
RAW.mkdir(exist_ok=True)
W, H = 1280, 720


class Camera:
    """RGB frames from the viewport camera of a plain SimulationContext scene."""

    def __init__(self):
        import omni.replicator.core as rep
        rp = rep.create.render_product("/OmniverseKit_Persp", (W, H))
        self.annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annot.attach([rp])

    def grab(self, sim, warmup=3):
        for _ in range(warmup):
            sim.render()
        return self.annot.get_data()[:, :, :3]


def save(img, name):
    imageio.imwrite(RAW / f"{name}.png", img)
    print("saved", name)


def scene_basics(sim):
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))


# ------------------------------------------------------------------ plain scenes

def drop():
    from isaaclab.assets import RigidObject, RigidObjectCfg
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=(3.2, 0.0, 0.9), target=(0.0, 0.0, 0.55))
    scene_basics(sim)
    balls = []
    for i, e in enumerate([0.0, 0.5, 0.9]):
        cfg = RigidObjectCfg(
            prim_path=f"/World/ball_{i}",
            spawn=sim_utils.SphereCfg(radius=0.1, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                      mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                      collision_props=sim_utils.CollisionPropertiesCfg(),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(restitution=e,
                                                                                      restitution_combine_mode="max"),
                                      visual_material=sim_utils.PreviewSurfaceCfg(
                                          diffuse_color=[(0.8, 0.2, 0.2), (0.9, 0.7, 0.1), (0.2, 0.6, 0.9)][i])),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5 * (i - 1), 1.1)))
        balls.append(RigidObject(cfg))
    sim.reset()
    cam = Camera()
    save(cam.grab(sim, 10), "drop_0")
    for k in range(int(0.8 / 0.005)):     # about the top of the first bounce
        sim.step(render=False)
    save(cam.grab(sim), "drop_1")


def _go2_robot(path, kp=25.0, pos=(0.0, 0.0, 0.4)):
    from isaaclab.assets import Articulation
    from rl_shortcut.go2_env import GO2_CFG
    cfg = GO2_CFG.replace(prim_path=path)
    cfg.actuators["legs"].stiffness = kp
    cfg.init_state.pos = pos
    return Articulation(cfg)


def _hold(robots, sim, seconds):
    for _ in range(int(seconds / 0.005)):
        for r in robots:
            r.set_joint_position_target(r.data.default_joint_pos)
            r.write_data_to_sim()
        sim.step(render=False)
        for r in robots:
            r.update(0.005)


def _to_default(robots):
    for r in robots:
        r.write_root_pose_to_sim(r.data.default_root_state[:, :7])
        r.write_joint_state_to_sim(r.data.default_joint_pos, r.data.default_joint_vel)


def stand():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=(2.2, 0.0, 0.7), target=(0.0, 0.0, 0.2))
    scene_basics(sim)
    robots = [_go2_robot(f"/World/Go2_{i}", kp, (0.0, y, 0.4)) for i, (kp, y) in enumerate([(25, -0.9), (5, 0.0), (0, 0.9)])]
    sim.reset()
    _to_default(robots)
    _hold(robots, sim, 2.0)
    save(Camera().grab(sim, 10), "stand")


def go2():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=(0.55, -0.95, 0.45), target=(0.0, 0.0, 0.22))
    scene_basics(sim)
    robots = [_go2_robot("/World/Go2")]
    sim.reset()
    _to_default(robots)
    _hold(robots, sim, 1.5)
    save(Camera().grab(sim, 10), "go2")


def parallel():
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from rl_shortcut.go2_env import GO2_CFG

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
        robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=(5.5, 5.5, 3.2), target=(0.0, 0.0, 0.0))
    scene = InteractiveScene(SceneCfg(num_envs=16, env_spacing=1.5))
    sim.reset()
    robot = scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    for _ in range(300):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(0.005)
    save(Camera().grab(sim, 10), "parallel")


def g1_ref(motion):
    from isaaclab.assets import Articulation
    from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG
    from rl_shortcut.motion import Motion
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    scene_basics(sim)
    robot = Articulation(G1_MINIMAL_CFG.replace(prim_path="/World/G1"))
    sim.reset()
    m = Motion(motion, sim.device)
    ids = m.joint_ids(robot.data.joint_names)
    cam = Camera()
    times = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0] if motion == "walk" else [0.0, 1.5, 3.5, 5.5, 7.5, 9.5]
    for k, t in enumerate(times):
        f = int(t * m.fps) % m.T
        jp = robot.data.default_joint_pos.clone()
        jp[:, ids] = m.dof[f]
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.write_root_pose_to_sim(torch.cat([m.root_pos[f], m.root_quat[f]]).unsqueeze(0))
        r = m.root_pos[f].tolist()
        if motion == "walk":
            sim.set_camera_view(eye=(r[0] + 0.3, r[1] - 3.2, 1.0), target=(r[0], r[1], 0.7))
        else:
            sim.set_camera_view(eye=(r[0] + 2.2, r[1] - 2.2, 1.2), target=(r[0], r[1], 0.6))
        sim.step(render=False)
        save(cam.grab(sim, 5), f"g1_ref_{motion}_{k}")
    if motion == "walk":                 # one standing G1 for the robot page
        jp = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.write_root_pose_to_sim(torch.tensor([[0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0]], device=sim.device))
        sim.set_camera_view(eye=(1.6, -1.6, 1.0), target=(0.0, 0.0, 0.6))
        sim.step(render=False)
        save(cam.grab(sim, 5), "g1")


# ------------------------------------------------------------------ trained policies (same as play.py)

def run_policy(task, motion, checkpoint, num_envs, seconds, grab_at, name, cmd=None, wide=False):
    import importlib

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from rl_shortcut.ppo import latest_checkpoint
    ns = argparse.Namespace(motion=motion)
    mod = importlib.import_module(f"rl_shortcut.{task}_env")
    env_cfg, runner_cfg = mod.make_env_cfg(ns), mod.make_runner_cfg(ns)
    env_cfg.scene.num_envs = num_envs
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.viewer.resolution = (W, H)
    if cmd is not None:
        c = env_cfg.commands.base_velocity
        c.rel_standing_envs = 0.0
        c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = [(v, v) for v in cmd]
    if task == "go2" and not wide:
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (0.6, -2.2, 0.7), (0.3, 0.0, 0.2)
        env_cfg.commands.base_velocity.debug_vis = False      # clean side view of the gait
    if task == "g1":
        env_cfg.viewer.eye, env_cfg.viewer.lookat = (0.8, -3.2, 0.0), (0.0, 0.0, -0.15)   # offsets from the pelvis
    if wide:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye, env_cfg.viewer.lookat = wide
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(str(latest_checkpoint(checkpoint)), load_optimizer=False)
    policy = runner.get_inference_policy(device="cuda:0")
    for _ in range(3):                 # the first frames of a new render product are empty
        env.render()
    obs = wrapped.get_observations()
    grab_steps = {int(t / env.step_dt): k for k, t in enumerate(grab_at)}
    for step in range(int(seconds / env.step_dt) + 1):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs))
        if step in grab_steps:
            env.render()
            save(env.render(), f"{name}_{grab_steps[step]}")
    env.close()


SHOTS = {
    "drop": drop, "stand": stand, "go2": go2, "parallel": parallel,
    "g1_ref_walk": lambda: g1_ref("walk"), "g1_ref_bruce_lee": lambda: g1_ref("bruce_lee"),
    "cartpole": lambda: run_policy("cartpole", "walk", "cartpole", 16, 3.0, [2.0], "cartpole",
                                   wide=((9.0, 9.0, 6.0), (0.0, 0.0, 1.5))),
    "go2_play": lambda: run_policy("go2", "walk", "go2", 16, 6.0, [5.0], "go2_play",
                                   wide=((7.5, 7.5, 5.0), (0.0, 0.0, 0.0))),
    "go2_forward": lambda: run_policy("go2", "walk", "go2", 1, 4.0, [1.0, 1.4, 1.8, 2.2], "go2_forward",
                                      cmd=(1.0, 0.0, 0.0)),
    "g1_play_walk": lambda: run_policy("g1", "walk", "g1_walk", 1, 6.0, [1, 2, 3, 4, 5, 6], "g1_play_walk"),
    "g1_walk_on_bruce_lee": lambda: run_policy("g1", "bruce_lee", "g1_walk", 1, 2.0, [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
                                               "g1_walk_on_bruce_lee"),
    "g1_play_bruce_lee": lambda: run_policy("g1", "bruce_lee", "g1_bruce_lee", 1, 10.0, [1.5, 3.5, 5.5, 7.5, 9.5],
                                            "g1_play_bruce_lee"),
}

if __name__ == "__main__":
    SHOTS[args.shot]()
    shutdown(app)
