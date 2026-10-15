"""PinkLAB rotary inverted pendulum kit (Furuta pendulum): balance the pendulum upright by turning the arm.

The robot is assets/pinklab_pendulum/usd/model_v3.usd, converted from the kit's URDF (page 4).
  base_to_arm      motor, turns the arm about the vertical axis (+-135 deg, 0.14 Nm)
  arm_to_pendulum  no motor, the pendulum swings freely (0 = hanging down, pi = upright)

The five parts of a manager-based RL env:
  actions (what the policy outputs), observations (what it sees), events (resets),
  rewards (what is good), terminations (when an episode ends).
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
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from rl_shortcut import ROOT
from rl_shortcut.ppo import make_ppo_cfg

EXPERIMENT = "pendulum"
PENDULUM_USD = str(ROOT / "assets/pinklab_pendulum/usd/model_v3.usd")
ARM = SceneEntityCfg("robot", joint_names=["base_to_arm"])
POLE = SceneEntityCfg("robot", joint_names=["arm_to_pendulum"])
MAX_TORQUE = 0.14       # Nm, the kit motor (URDF effort limit)

PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PENDULUM_USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(solver_position_iteration_count=8),
    ),
    # the default pose is the goal: arm in the middle, pendulum upright
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"base_to_arm": 0.0, "arm_to_pendulum": math.pi}),
    actuators={
        # torque control: no position gains, the policy's torque goes straight to the arm
        "arm_motor": ImplicitActuatorCfg(joint_names_expr=["base_to_arm"], stiffness=0.0, damping=0.0,
                                         effort_limit_sim=MAX_TORQUE),
        # no motor on the pendulum; only the bearing damping from the URDF
        "pendulum": ImplicitActuatorCfg(joint_names_expr=["arm_to_pendulum"], stiffness=0.0, damping=3.16e-5,
                                        effort_limit_sim=0.0),
    },
)


def angle_sq(env, asset_cfg):
    """Joint angle from the default pose, squared: 0 at the goal, grows away from it."""
    robot = env.scene[asset_cfg.name]
    angle = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class ActionsCfg:
    # one number per env: arm motor torque, action 1.0 = 0.14 Nm (the motor's limit)
    arm_torque = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["base_to_arm"], scale=MAX_TORQUE)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)    # arm angle, pendulum angle from upright (2)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)    # arm and pendulum angular velocity (2)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_arm = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                          params={"asset_cfg": ARM, "position_range": (-0.5, 0.5), "velocity_range": (-1.0, 1.0)})
    reset_pole = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                           params={"asset_cfg": POLE, "position_range": (-0.2, 0.2), "velocity_range": (-1.0, 1.0)})


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_upright = RewTerm(func=angle_sq, weight=-1.0, params={"asset_cfg": POLE})
    arm_center = RewTerm(func=angle_sq, weight=-0.1, params={"asset_cfg": ARM})
    arm_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.01, params={"asset_cfg": ARM})
    pole_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.005, params={"asset_cfg": POLE})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pole_fell = DoneTerm(func=mdp.joint_pos_out_of_manual_limit,
                         params={"asset_cfg": POLE, "bounds": (math.pi - 0.8, math.pi + 0.8)})
    arm_out = DoneTerm(func=mdp.joint_pos_out_of_manual_limit, params={"asset_cfg": ARM, "bounds": (-2.0, 2.0)})


@configclass
class PendulumEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=0.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005               # physics 200 Hz
        self.decimation = 2               # policy 100 Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0
        self.viewer.eye = (0.6, 0.6, 0.45)
        self.viewer.lookat = (0.0, 0.0, 0.12)


def make_env_cfg(args):
    return PendulumEnvCfg()


def make_runner_cfg(args):
    return make_ppo_cfg(EXPERIMENT, max_iterations=150, hidden=(32, 32), steps_per_env=16, entropy_coef=0.01)
