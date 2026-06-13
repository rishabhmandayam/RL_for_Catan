"""
Example script to run a heads-up experiment between two Catan bots.

To run this:
    python -m sandbox.example_experiment
"""

from catanatron.models.player import RandomPlayer
from sandbox.bots.heuristics.smart_bot import SmartBot
from sandbox.experiments.run_experiment import run_experiment

# NOTE: For multiprocessing, builder functions must be picklable (i.e., defined at module level).
def build_bot0(color):
    """Builder for Player 0."""
    return SmartBot(color, epsilon=0.05)

def build_bot1(color):
    """Builder for Player 1."""
    return RandomPlayer(color)

if __name__ == "__main__":
    # Run a quick experiment with 10 games
    run_experiment(
        name="smart_vs_random",
        bot0_builder=build_bot0,
        bot1_builder=build_bot1,
        num_games=10,
        num_workers=4  # Set to None to use all available cores
    )
