"""
Training script for training a Maskable PPO agent on Catanatron using Stable Baselines 3.

To run this, make sure to add stable-baselines3 and sb3-contrib dependencies:
    uv add stable-baselines3 sb3-contrib
    uv run python -m sandbox.bots.rl.train_ppo
"""

import os
import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym

# 1. Action Masking Function
def mask_fn(env: gym.Env) -> np.ndarray:
    """Retrieves valid action mask from the unwrapped environment."""
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])

def make_env(opponent_player):
    """Helper to instantiate and wrap the Catan env."""
    # We configure a 1v1 heads-up game. 
    # The RL agent will play as Color.BLUE, and the opponent plays as another color.
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

def main():
    #1. Create directory for saving models and logs
    models_dir = "sandbox/bots/rl/models"
    log_dir = "sandbox/bots/rl/logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Configure the Opponent
    opponent = WeightedRandomPlayer(Color.RED)
    
    # 3. Instantiate Environments
    num_cpu = 8
    train_env = make_env(opponent)
    train_env = Monitor(train_env, log_dir) # Wrap with Monitor to track episodic stats (rewards, lengths)

    eval_env = make_env(opponent)
    eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))

    # 4. Instantiate the Maskable PPO model
    # We use MaskableActorCriticPolicy which supports action masks.
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Entropy coefficient encourages exploration
        verbose=1,
        tensorboard_log=log_dir
    )

    # 5. Callbacks
    # Periodically evaluates the agent and saves the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=log_dir,
        eval_freq=10_000,      # Evaluate every 10k steps
        deterministic=True,
        render=False
    )
    
    # Save a checkpoint model every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(models_dir, "checkpoints"),
        name_prefix="ppo_catan"
    )

    # 6. Train the Agent
    total_timesteps = 100_000
    print(f"Starting training for {total_timesteps} timesteps against {opponent.__class__.__name__}...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )

    # Save final model
    final_model_path = os.path.join(models_dir, "ppo_catan_final")
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
