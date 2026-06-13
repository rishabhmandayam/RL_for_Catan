from catanatron.game import GameAccumulator
from catanatron.state_functions import player_key

class ExperimentAccumulator(GameAccumulator):
    """
    Accumulator that tracks metrics for a single game and stores them in a dict.
    """
    def __init__(self, p0_color, p1_color):
        self.p0_color = p0_color
        self.p1_color = p1_color
        self.metrics = {
            "p0_vps": 0,
            "p1_vps": 0,
            "p0_actual_vps": 0,
            "p1_actual_vps": 0,
            "p0_longest_road": False,
            "p1_longest_road": False,
            "p0_largest_army": False,
            "p1_largest_army": False,
            "turns": 0,
            "winning_color": None,
            "winner": None, # "p0", "p1", or "tie"
        }

    def after(self, game):
        """Called when the game finishes."""
        winning_color = game.winning_color()
        if winning_color == self.p0_color:
            self.metrics["winner"] = "p0"
        elif winning_color == self.p1_color:
            self.metrics["winner"] = "p1"
        else:
            self.metrics["winner"] = "tie"
            
        self.metrics["winning_color"] = str(winning_color)
        self.metrics["turns"] = game.state.num_turns
        
        p0_key = player_key(game.state, self.p0_color)
        p1_key = player_key(game.state, self.p1_color)
        
        self.metrics["p0_vps"] = game.state.player_state[f"{p0_key}_VICTORY_POINTS"]
        self.metrics["p1_vps"] = game.state.player_state[f"{p1_key}_VICTORY_POINTS"]
        self.metrics["p0_actual_vps"] = game.state.player_state[f"{p0_key}_ACTUAL_VICTORY_POINTS"]
        self.metrics["p1_actual_vps"] = game.state.player_state[f"{p1_key}_ACTUAL_VICTORY_POINTS"]
        
        self.metrics["p0_longest_road"] = game.state.player_state[f"{p0_key}_HAS_ROAD"]
        self.metrics["p1_longest_road"] = game.state.player_state[f"{p1_key}_HAS_ROAD"]
        self.metrics["p0_largest_army"] = game.state.player_state[f"{p0_key}_HAS_ARMY"]
        self.metrics["p1_largest_army"] = game.state.player_state[f"{p1_key}_HAS_ARMY"]
