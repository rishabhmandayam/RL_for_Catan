import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron.models.player import Player
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.envs.action_space import (
    get_action_array,
    from_action_space,
    to_action_space,
)

class SB3Player(Player):
    """
    A Catanatron Player that uses a trained Stable Baselines 3 model 
    (with Action Masking) to decide its moves.
    """

    def __init__(self, color, model_path: str, is_bot: bool = True):
        super().__init__(color, is_bot)
        # Load the Maskable PPO model
        self.model = MaskablePPO.load(model_path)
        self.map_type = None
        self.features = None
        self.player_colors = None
        self.action_array = None

    def _init_spaces(self, game):
        # Lazy initialization based on the game's actual map and players
        if self.map_type is None:
            self.map_type = game.state.board.map.map_type
            
            # Catanatron's Gym Env always puts the 'ego' player first
            enemy_colors = [c for c in game.state.colors if c != self.color]
            self.player_colors = tuple([self.color] + enemy_colors)
            
            num_players = len(game.state.colors)
            self.features = get_feature_ordering(num_players, self.map_type)
            self.action_array = get_action_array(self.player_colors, self.map_type)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        self._init_spaces(game)

        # 1. Build observation
        sample = create_sample(game, self.color)
        obs = np.array([sample[f] for f in self.features], dtype=np.float32)

        # 2. Build action mask
        valid_actions_idx = [
            to_action_space(a, self.player_colors, self.map_type)
            for a in playable_actions
        ]
        action_mask = np.zeros(len(self.action_array), dtype=np.float32)
        for idx in valid_actions_idx:
            action_mask[idx] = 1
            
        action_mask_bool = np.array([bool(i) for i in action_mask])

        # 3. Predict with SB3 model
        action_idx, _states = self.model.predict(
            obs, 
            action_masks=action_mask_bool,
            deterministic=True
        )

        # 4. Convert back to Catanatron Action
        return from_action_space(int(action_idx), self.color, self.player_colors, self.map_type)

    def __str__(self):
        return f"SB3Player({self.color})"
