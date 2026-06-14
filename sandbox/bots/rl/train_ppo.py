"""
Training script for training a Maskable PPO agent on Catanatron using Stable Baselines 3.

To run this, make sure to add dependencies:
    uv add stable-baselines3 sb3-contrib
    uv run python -m sandbox.bots.rl.train_ppo
"""

import os
import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from catanatron.models.player import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym


class CatanMetricsCallback(BaseCallback):
    """
    Custom callback to log Catan-specific metrics (like win rate and turns)
    to TensorBoard from the Monitor wrapper.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if an episode just finished
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            if "episode" in info:
                self.logger.record("catan/game_reward", info["episode"]["r"])
                self.logger.record("catan/game_turns", info["episode"]["l"])
                
                # In simple_reward, 1 means win.
                is_win = 1.0 if info["episode"]["r"] > 0 else 0.0
                self.logger.record("catan/win_rate_rolling", is_win)
        return True


def mask_fn(env: gym.Env) -> np.ndarray:
    """Retrieves valid action mask from the unwrapped environment."""
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])


def make_env(opponent_player):
    """Helper to instantiate and wrap the Catan env."""
    def _init() -> gym.Env:
        env = gym.make(
            "catanatron/Catanatron-v0",
            config={
                "map_type": "BASE",
                "vps_to_win": 10,
                "enemies": [opponent_player],
                "representation": "vector",
            }
        )
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def main():
    models_dir = "sandbox/bots/rl/models"
    log_dir = "sandbox/bots/rl/logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "total_timesteps": 1_000_000,
    }

    # 1. Configure the Opponent & Environments
    opponent = WeightedRandomPlayer(Color.RED)
    num_cpu = 8
    
    env_fns = [make_env(opponent) for _ in range(num_cpu)]

    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env, log_dir)

    eval_env = DummyVecEnv([make_env(opponent)])
    eval_env = VecMonitor(eval_env, os.path.join(log_dir, "eval"))

    # 2. Instantiate the Maskable PPO model
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        train_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        verbose=1,
        tensorboard_log=log_dir  # Logs straight to TensorBoard
    )

    # 3. Set up Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=log_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(models_dir, "checkpoints"),
        name_prefix="ppo_catan"
    )

    catan_metrics_callback = CatanMetricsCallback()

    # 4. Train!
    print(f"Starting {num_cpu}-core parallel training for {config['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[
            checkpoint_callback, 
            catan_metrics_callback
        ],
        progress_bar=True
    )

    # Save final model
    final_model_path = os.path.join(models_dir, "ppo_catan_final")
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
