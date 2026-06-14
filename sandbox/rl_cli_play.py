import os
import sys

# Ensure root of repository is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catanatron.cli import register_cli_player
from sandbox.bots.rl.sb3_bot import SB3Player

class RLPlayerWrapper(SB3Player):
    """Stable Baselines 3 PPO RL Player"""
    def __init__(self, color):
        model_path = os.path.join(
            os.path.dirname(__file__),
            "bots/rl/models/best_model/best_model.zip"
        )
        if not os.path.exists(model_path):
            model_path = os.path.join(
                os.path.dirname(__file__),
                "bots/rl/models/ppo_catan_final.zip"
            )
        super().__init__(color, model_path)

register_cli_player("RL", RLPlayerWrapper)
