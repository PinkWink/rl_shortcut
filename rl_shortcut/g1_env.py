"""Unitree G1 humanoid imitating a reference motion (walk or Bruce Lee kung fu).

One env for every clip: the observation always has the same 107 numbers, so a policy
trained on one clip can be run on another clip (page 9: the walking policy tries kung fu).
"""
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

from rl_shortcut.motion import Motion, quat_yaw
from rl_shortcut.ppo import make_ppo_cfg

# the 23 body joints the policy drives (the 14 finger joints keep their default pose)
BODY_JOINTS = [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint",
               ".*_ankle_pitch_joint", ".*_ankle_roll_joint", "torso_joint",
               ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
               ".*_elbow_pitch_joint", ".*_elbow_roll_joint"]
BODY = SceneEntityCfg("robot", joint_names=BODY_JOINTS)
ARM_SLICE = slice(13, 23)    # arm joints inside the clip's 23


# ---------------------------------------------------------------- reference motion per env

class Reference:
    """The clip plus each env's starting frame (set by the reset event)."""

    def __init__(self, env):
        self.motion = Motion(env.cfg.motion, env.device)
        robot = env.scene["robot"]
        self.ids = torch.tensor(self.motion.joint_ids(robot.data.joint_names), device=env.device)
        self.start = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def frame(self, env):
        t = env.episode_length_buf * env.step_dt
        return (self.start + (t * self.motion.fps).long()) % self.motion.T


def ref(env):
    if not hasattr(env, "_reference"):
        env._reference = Reference(env)
    return env._reference


def _yaw_error(env):
    r = ref(env)
    return quat_yaw(env.scene["robot"].data.root_quat_w) - r.motion.yaw[r.frame(env)]


# ---------------------------------------------------------------- observations

def ref_joint_pos(env):
    """Target joint angles now, relative to the default pose (23)."""
    r = ref(env)
    return r.motion.dof[r.frame(env)] - env.scene["robot"].data.default_joint_pos[:, r.ids]


def ref_phase(env):
    """Where we are in the clip, as sin/cos (2)."""
    r = ref(env)
    phase = r.frame(env).float() / r.motion.T * 2 * torch.pi
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


def yaw_error(env):
    """Heading error to the clip, as sin/cos (2)."""
    e = _yaw_error(env)
    return torch.stack([torch.sin(e), torch.cos(e)], dim=1)


def ref_velocity(env):
    """Target ground velocity in the robot's heading frame (2)."""
    r = ref(env)
    v = r.motion.root_vel[r.frame(env), :2]
    yaw = quat_yaw(env.scene["robot"].data.root_quat_w)
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], dim=1)


# ---------------------------------------------------------------- rewards

def track_joints(env, scale=1.0, arms_only=False):
    r = ref(env)
    target = r.motion.dof[r.frame(env)]
    q = env.scene["robot"].data.joint_pos[:, r.ids]
    if arms_only:
        target, q = target[:, ARM_SLICE], q[:, ARM_SLICE]
    return torch.exp(-scale * torch.sum(torch.square(q - target), dim=1))


def track_velocity(env):
    r = ref(env)
    v = env.scene["robot"].data.root_lin_vel_w[:, :2]
    return torch.exp(-4.0 * torch.sum(torch.square(v - r.motion.root_vel[r.frame(env), :2]), dim=1))


def track_yaw(env):
    e = _yaw_error(env)
    return torch.exp(-4.0 * torch.square(torch.atan2(torch.sin(e), torch.cos(e))))


def track_height(env):
    r = ref(env)
    z = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-torch.square(z - r.motion.root_pos[r.frame(env), 2]) / 0.01)


# ---------------------------------------------------------------- termination and reset

def height_off(env, limit=0.3):
    """Pelvis more than `limit` m above or below the clip: fallen (or jumped)."""
    r = ref(env)
    return (env.scene["robot"].data.root_pos_w[:, 2] - r.motion.root_pos[r.frame(env), 2]).abs() > limit


def reset_to_clip(env, env_ids):
    """Reference State Initialization: start each episode at a random frame of the clip."""
    r, robot, m = ref(env), env.scene["robot"], ref(env).motion
    f = torch.randint(0, m.T, (len(env_ids),), device=env.device)
    r.start[env_ids] = f
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos[:, r.ids] = m.dof[f]
    joint_vel[:, r.ids] = m.dof_vel[f]
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root = robot.data.default_root_state[env_ids].clone()
    root[:, 0:2] = env.scene.env_origins[env_ids, 0:2]
    root[:, 2] = m.root_pos[f, 2]
    root[:, 3:7] = m.root_quat[f]
    root[:, 7:10] = m.root_vel[f]
    root[:, 10:12] = 0.0
    root[:, 12] = m.yaw_rate[f]
    robot.write_root_pose_to_sim(root[:, 0:7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)


# ---------------------------------------------------------------- env

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
                                      history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=BODY_JOINTS, scale=0.25)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)                                  # 3
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)                                  # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity)                        # 3
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": BODY})        # 23
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": BODY})        # 23
        actions = ObsTerm(func=mdp.last_action)                                        # 23
        ref_joint_pos = ObsTerm(func=ref_joint_pos)                                    # 23
        ref_phase = ObsTerm(func=ref_phase)                                            # 2
        yaw_error = ObsTerm(func=yaw_error)                                            # 2
        ref_velocity = ObsTerm(func=ref_velocity)                                      # 2

        def __post_init__(self):
            self.concatenate_terms = True     # 107 numbers

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset = EventTerm(func=reset_to_clip, mode="reset")


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    joints = RewTerm(func=track_joints, weight=5.0)                                     # whole-body pose
    arms = RewTerm(func=track_joints, weight=2.0, params={"scale": 3.0, "arms_only": True})
    velocity = RewTerm(func=track_velocity, weight=2.0)
    yaw = RewTerm(func=track_yaw, weight=2.0)
    height = RewTerm(func=track_height, weight=2.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    feet_slide = RewTerm(func=loco.feet_slide, weight=-0.1, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tilted = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    height_off = DoneTerm(func=height_off, params={"limit": 0.3})


@configclass
class G1EnvCfg(ManagerBasedRLEnvCfg):
    motion: str = "walk"              # motions/<motion>.npz
    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005               # physics 200 Hz
        self.decimation = 4               # policy 50 Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.0, 3.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.7)


def make_env_cfg(args):
    cfg = G1EnvCfg()
    cfg.motion = args.motion
    return cfg


def make_runner_cfg(args):
    iterations = {"walk": 2000, "bruce_lee": 3000}.get(args.motion, 3000)
    return make_ppo_cfg(f"g1_{args.motion}", max_iterations=iterations, hidden=(256, 128, 64),
                        entropy_coef=0.008, init_noise_std=0.5)
