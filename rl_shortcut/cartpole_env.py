"""Inverted pendulum on a cart (cartpole): the smallest complete RL environment.

The five parts of a manager-based RL env:
  actions (what the policy outputs), observations (what it sees), events (resets),
  rewards (what is good), terminations (when an episode ends).
"""
import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
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
from isaaclab.utils import configclass
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from rl_shortcut.ppo import make_ppo_cfg

EXPERIMENT = "cartpole"
CART = SceneEntityCfg("robot", joint_names=["slider_to_cart"])
POLE = SceneEntityCfg("robot", joint_names=["cart_to_pole"])


def pole_angle_sq(env, asset_cfg):
    """Pole angle squared: 0 when upright, grows as it tips over."""
    angle = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class ActionsCfg:
    # one number per env: force on the cart, action 1.0 = 100 N
    cart_force = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)    # cart position, pole angle (2)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)    # cart velocity, pole angular velocity (2)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_cart = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                           params={"asset_cfg": CART, "position_range": (-1.0, 1.0), "velocity_range": (-0.5, 0.5)})
    reset_pole = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                           params={"asset_cfg": POLE, "position_range": (-0.25 * math.pi, 0.25 * math.pi),
                                   "velocity_range": (-0.25 * math.pi, 0.25 * math.pi)})


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_upright = RewTerm(func=pole_angle_sq, weight=-1.0, params={"asset_cfg": POLE})
    cart_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.01, params={"asset_cfg": CART})
    pole_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.005, params={"asset_cfg": POLE})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out = DoneTerm(func=mdp.joint_pos_out_of_manual_limit, params={"asset_cfg": CART, "bounds": (-3.0, 3.0)})


@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 1 / 120             # physics 120 Hz
        self.decimation = 2               # policy 60 Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0
        self.viewer.eye = (8.0, 0.0, 5.0)


def make_env_cfg(args):
    return CartpoleEnvCfg()


def make_runner_cfg(args):
    return make_ppo_cfg(EXPERIMENT, max_iterations=150, hidden=(32, 32), steps_per_env=16)
