"""Unitree Go2 walking on flat ground, following a velocity command (vx, vy, yaw rate).

The robot comes from go2.usd on the Isaac Lab asset server; GO2_CFG below adds the
motor model and the standing pose (page 6). The env follows Isaac Lab's flat Go2 task.
"""
import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco
from isaaclab.actuators import DCMotorCfg
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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from rl_shortcut.ppo import make_ppo_cfg

EXPERIMENT = "go2"
GO2_USD = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"

# standing pose: hips slightly out, thighs forward, knees bent (rad)
STAND_POSE = {".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
              "F[L,R]_thigh_joint": 0.8, "R[L,R]_thigh_joint": 1.0, ".*_calf_joint": -1.5}

GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_USD,
        activate_contact_sensors=True,
        # slow push-out when bodies overlap (otherwise an overlap at spawn launches the robot)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.4), joint_pos=STAND_POSE),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 12 leg motors: PD position control, torque limited by a DC motor curve (23.5 Nm, 30 rad/s)
        "legs": DCMotorCfg(joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                           stiffness=25.0, damping=0.5, effort_limit=23.5, saturation_effort=23.5,
                           velocity_limit=30.0, friction=0.0),
    },
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)))
    robot: ArticulationCfg = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # contact forces on every body: feet for air time, base for "fell over"
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0), rel_standing_envs=0.02,
        heading_command=False, debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
                                                    ang_vel_z=(-1.0, 1.0)),
    )


@configclass
class ActionsCfg:
    # 12 joint targets = standing pose + 0.25 * action
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25,
                                           use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))           # 3
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))           # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 3
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))         # 12
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))           # 12
        actions = ObsTerm(func=mdp.last_action)                                                     # 12

        def __post_init__(self):
            self.enable_corruption = True     # sensor noise while training
            self.concatenate_terms = True     # 48 numbers

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    friction = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                         params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                                 "static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6),
                                 "restitution_range": (0.0, 0.0), "num_buckets": 64})
    base_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup",
                          params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                                  "mass_distribution_params": (-1.0, 3.0), "operation": "add"})
    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                           params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
                                   "velocity_range": {}})
    reset_joints = EventTerm(func=mdp.reset_joints_by_scale, mode="reset",
                             params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)})


@configclass
class RewardsCfg:
    # task: follow the command
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5,
                               params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.75,
                              params={"command_name": "base_velocity", "std": 0.5})
    feet_air_time = RewTerm(func=loco.feet_air_time, weight=0.25,
                            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                                    "command_name": "base_velocity", "threshold": 0.5})
    # penalties: keep the body steady and the motors gentle
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact,
                            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
                                    "threshold": 1.0})


@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005               # physics 200 Hz
        self.decimation = 4               # policy 50 Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


def make_env_cfg(args):
    return Go2EnvCfg()


def make_runner_cfg(args):
    return make_ppo_cfg(EXPERIMENT, max_iterations=300, hidden=(128, 128, 128), steps_per_env=24,
                        entropy_coef=0.01)
