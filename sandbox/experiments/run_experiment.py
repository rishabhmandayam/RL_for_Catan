import os
import json
import uuid
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from catanatron.game import Game
from catanatron.models.player import Color
from sandbox.experiments.accumulators import ExperimentAccumulator

def play_one_game(bot0_builder, bot1_builder, seed=None):
    """
    Plays a single heads-up game.
    bot_builder: callable that takes a Color and returns a Player.
    """
    p0_color = Color.BLUE
    p1_color = Color.RED
    
    bot0 = bot0_builder(p0_color)
    bot1 = bot1_builder(p1_color)
    
    game = Game(players=[bot0, bot1], seed=seed, vps_to_win=10)
    acc = ExperimentAccumulator(p0_color, p1_color)
    
    winning_color = game.play(accumulators=[acc])
    return acc.metrics

def run_experiment(name: str, bot0_builder, bot1_builder, num_games: int = 100, num_workers: int = None):
    """
    Runs `num_games` 1v1 matches between bot0 and bot1 using multiple processes.
    Saves the results down to sandbox/results/<name>_<timestamp>.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"sandbox/results/{name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Starting experiment: {name} ({num_games} games)")
    print(f"Saving results to: {exp_dir}")
    
    results = []
    
    # We pass the builder functions which should be picklable or defined at module level.
    # If the user uses lambda, it might fail. Better to use functools.partial or standard defs.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_games):
            seed = int(uuid.uuid4().int % (2**32))
            futures.append(executor.submit(play_one_game, bot0_builder, bot1_builder, seed))
            
        for future in tqdm(as_completed(futures), total=num_games, desc="Games"):
            try:
                metrics = future.result()
                results.append(metrics)
            except Exception as e:
                print(f"Game failed with exception: {e}")
                
    # Save raw results as JSON Lines
    games_file = os.path.join(exp_dir, "games.jsonl")
    with open(games_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    # Compute Summary Stats
    if results:
        p0_wins = sum(1 for r in results if r["winner"] == "p0")
        p1_wins = sum(1 for r in results if r["winner"] == "p1")
        ties = sum(1 for r in results if r["winner"] == "tie")
        
        avg_turns = sum(r["turns"] for r in results) / len(results)
        
        summary = {
            "num_games": len(results),
            "p0_win_rate": p0_wins / len(results),
            "p1_win_rate": p1_wins / len(results),
            "tie_rate": ties / len(results),
            "avg_turns": avg_turns,
        }
        
        summary_file = os.path.join(exp_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)
            
        print("\nExperiment Summary:")
        print(f"P0 Win Rate: {summary['p0_win_rate']:.2%}")
        print(f"P1 Win Rate: {summary['p1_win_rate']:.2%}")
        print(f"Avg Turns:   {summary['avg_turns']:.1f}")
        
    return exp_dir
