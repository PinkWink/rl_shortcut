"""PPO settings (RSL-RL) and checkpoint lookup shared by train.py and play.py."""
import re

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from rl_shortcut import CHECKPOINT_DIR, LOG_DIR


def make_ppo_cfg(experiment, max_iterations, hidden=(256, 128, 64), steps_per_env=24,
                 entropy_coef=0.005, init_noise_std=1.0, save_interval=50):
    return RslRlOnPolicyRunnerCfg(
        experiment_name=experiment,
        max_iterations=max_iterations,
        num_steps_per_env=steps_per_env,      # rollout length per env before each update
        save_interval=save_interval,
        obs_groups={"policy": ["policy"]},
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=init_noise_std,     # exploration noise at the start
            actor_hidden_dims=list(hidden),
            critic_hidden_dims=list(hidden),
            activation="elu",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=entropy_coef, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=1.0e-3, schedule="adaptive", desired_kl=0.01,
            gamma=0.99, lam=0.95, max_grad_norm=1.0,
        ),
    )


def latest_checkpoint(experiment):
    """Newest model_<iter>.pt of the newest run in logs/<experiment>/, else checkpoints/<experiment>.pt."""
    runs = sorted((LOG_DIR / experiment).glob("*/model_*.pt"), key=lambda p: (p.parent.name, _iteration(p)))
    if runs:
        return runs[-1]
    shipped = CHECKPOINT_DIR / f"{experiment}.pt"
    return shipped if shipped.exists() else None


def _iteration(path):
    return int(re.search(r"model_(\d+)\.pt$", path.name).group(1))
