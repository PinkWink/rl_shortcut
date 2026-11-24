"""Pinky Pro in the City of Pink: drive around the outer ring road, seeing only the road mask of its camera.

The robot is assets/pinky_pro/usd/pinky.usd (two drive wheels + a rear caster), the town is assets/pink_city/.
  actions       left and right wheel speed (2)
  observations  road view 8 x 16 (128) + last action (2); see pink_city.py
                  --perception map   from the floor label map (training: fast, no camera)
                  --perception yolo  from YOLO segmentation of the camera image (play; needs --enable_cameras)
  rewards       speed along the ring road, staying in the right lane
  terminations  left the road
"""
import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from rl_shortcut import ROOT, pink_city
from rl_shortcut.ppo import make_ppo_cfg

EXPERIMENT = "pinky"
PINKY_USD = str(ROOT / "assets/pinky_pro/usd/pinky.usd")
WHEELS = ["l_wheel_joint", "r_wheel_joint"]
WHEEL_RADIUS = 0.028
MAX_SPEED = 0.25                              # m/s
MAX_WHEEL = MAX_SPEED / WHEEL_RADIUS          # rad/s, action 1.0 = this wheel speed

PINKY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PINKY_USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(-pink_city.RING - pink_city.LANE, 0.0, 0.002)),
    actuators={
        # speed control: no position gain, the damping pulls the wheel speed to the target
        "wheels": ImplicitActuatorCfg(joint_names_expr=WHEELS, stiffness=0.0, damping=0.5, effort_limit_sim=0.3),
        "caster": ImplicitActuatorCfg(joint_names_expr=["caster_.*"], stiffness=0.0, damping=0.0),
    },
)


def city_pose(env):
    """Robot position in its own city (x, y) and heading (yaw)."""
    robot = env.scene["robot"]
    xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return xy, robot.data.heading_w


def road_map(env):
    if not hasattr(env, "_road_map"):
        env._road_map = pink_city.RoadMap(env.device)
    return env._road_map


def road_view(env, flip=0.0):
    """Road mask of the lower camera image as 8 x 16 cells. flip: share of cells replaced by noise (training only)."""
    if env.cfg.perception == "yolo":
        from rl_shortcut.road_seg import yolo_view
        return yolo_view(env)
    xy, yaw = city_pose(env)
    view = road_map(env).view(xy, yaw)
    if flip > 0.0 and env.cfg.observations.policy.enable_corruption:
        noisy = torch.rand_like(view) < flip
        view = torch.where(noisy, torch.rand_like(view), view)
    return view


def lane_error(env):
    """Distance [m] from the middle of the right lane (+ = towards the outside)."""
    xy, _ = city_pose(env)
    return pink_city.ring_distance(xy)[0] - pink_city.LANE


def progress(env):
    """Speed [m/s] along the ring road (anticlockwise)."""
    xy, _ = city_pose(env)
    return (env.scene["robot"].data.root_lin_vel_w[:, :2] * pink_city.ring_distance(xy)[1]).sum(dim=1)


def in_lane(env, std=0.05):
    return torch.exp(-torch.square(lane_error(env) / std))


def off_road(env, margin=0.04):
    """The robot centre is closer than `margin` to the edge of the ring road, or beyond it."""
    xy, _ = city_pose(env)
    return pink_city.ring_distance(xy)[0].abs() > pink_city.ROAD / 2 - margin


def reset_on_ring(env, env_ids, lane_noise=0.04, yaw_noise=0.4):
    """Start anywhere on the ring road, a little off the lane centre and a little turned."""
    robot, n = env.scene["robot"], len(env_ids)
    angle = torch.rand(n, device=env.device) * 2 * math.pi
    offset = pink_city.LANE + (torch.rand(n, device=env.device) * 2 - 1) * lane_noise
    xy, heading = pink_city.ring_point(angle, offset)
    yaw = heading + (torch.rand(n, device=env.device) * 2 - 1) * yaw_noise
    pose = robot.data.default_root_state[env_ids, :7].clone()
    pose[:, :2] = env.scene.env_origins[env_ids, :2] + xy
    pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    robot.write_root_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids], robot.data.default_joint_vel[env_ids],
                                   env_ids=env_ids)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    # training: only the painted floor (the road is a picture; nothing to collide with)
    city = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/City", spawn=sim_utils.UsdFileCfg(usd_path=pink_city.FLOOR_USD))
    robot: ArticulationCfg = PINKY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1200.0))
    sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=1500.0, angle=1.5),
                       init_state=AssetBaseCfg.InitialStateCfg(rot=(0.94, 0.29, 0.16, 0.05)))


# the robot's own camera, used only with --perception yolo (same place and lens as the real Pinky Pro)
CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_footprint/front_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(pink_city.CAM_X, 0.0, pink_city.CAM_HEIGHT),
                                    rot=(math.cos(pink_city.CAM_PITCH / 2), 0.0, math.sin(pink_city.CAM_PITCH / 2), 0.0),
                                    convention="world"),
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.1, horizontal_aperture=2 * 16.1 * math.tan(pink_city.CAM_HFOV / 2), clipping_range=(0.01, 20.0)),
    width=pink_city.IMAGE[0], height=pink_city.IMAGE[1], data_types=["rgb"],
)


@configclass
class ActionsCfg:
    # two numbers per robot: left and right wheel speed, action 1.0 = 0.25 m/s
    wheels = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=WHEELS, scale=MAX_WHEEL,
                                        clip={".*": (-MAX_WHEEL, MAX_WHEEL)})


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        road = ObsTerm(func=road_view, params={"flip": 0.03})     # 8 x 16 road view (128)
        actions = ObsTerm(func=mdp.last_action)                   # last wheel speeds (2)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True                         # play.py turns the noise off

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset = EventTerm(func=reset_on_ring, mode="reset")


@configclass
class RewardsCfg:
    progress = RewTerm(func=progress, weight=8.0)                 # go around, anticlockwise
    in_lane = RewTerm(func=in_lane, weight=1.0)                   # in the middle of the right lane
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)  # no jerky steering
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    off_road = DoneTerm(func=off_road)


@configclass
class PinkyEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1024, env_spacing=3.4)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    perception: str = "map"

    def __post_init__(self):
        self.sim.dt = 0.005               # physics 200 Hz
        self.decimation = 10              # policy 20 Hz (about what a camera + YOLO can do)
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.viewer.eye = (-2.0, -2.7, 2.4)
        self.viewer.lookat = (0.0, 0.05, 0.0)


def make_env_cfg(args):
    cfg = PinkyEnvCfg()
    cfg.perception = getattr(args, "perception", "map")
    if cfg.perception == "yolo" or getattr(args, "city", False):
        cfg.scene.city.spawn.usd_path = pink_city.CITY_USD          # walls and buildings too
    if cfg.perception == "yolo":
        cfg.scene.camera = CAMERA_CFG
    return cfg


def make_runner_cfg(args):
    return make_ppo_cfg(EXPERIMENT, max_iterations=300, hidden=(128, 64), steps_per_env=32, entropy_coef=0.005)
