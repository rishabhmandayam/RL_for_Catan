# Sandbox: Playing Against Catanatron Bots (RL and Minimax)

This sandbox directory contains experiments, custom heuristics, and reinforcement learning bots for Catanatron. 

This README explains how you can play interactive games against these bots, including the trained Reinforcement Learning (RL) bot.

---

## The Bots

1. **RL Bot (`RL` / `SB3Player`)**:
   - Uses a Stable Baselines 3 **Maskable PPO** policy network.
   - Loads the trained model weights from `sandbox/bots/rl/models/best_model/best_model.zip` (or `ppo_catan_final.zip`).
   - **Important Constraint**: The RL bot only supports **2-player games** (1 human + 1 RL bot). Because its observation features (shape `(614,)`) and action spaces (size `332`) are fixed to the 2-player setup it was trained on, running it in 3 or 4-player games will cause shape/index mismatches.
2. **Catanatron Bot (`CATANATRON` / `AlphaBetaPlayer`)**:
   - Uses an alpha-beta minimax look-ahead search evaluating leaf states with a handcrafted value function.
   - Supports 2, 3, and 4-player games perfectly out of the box.
3. **Weighted Random Bot (`WEIGHTED_RANDOM` / `WeightedRandomPlayer`)**:
   - Selects moves randomly but favors high-value actions (buying settlements, cities, development cards).
   - Supports 2, 3, and 4-player games.

---

## Method 1: Play in the Web UI (Recommended)

The RL bot and other bots are fully integrated into the web client and server.

### 1. Start the Docker Services
From the root of the repository, start the multi-container environment:
```bash
docker compose up
```
This launches the database, Flask web server (`localhost:5001`), and React frontend (`localhost:3000`).

### 2. Configure and Play
1. Open your browser and navigate to **http://localhost:3000**.
2. **To play against the RL Bot (2-Player)**:
   - Configure a 2-player game.
   - Set player 1 to **Human** and player 2 to **RL Bot (Stable Baselines)**.
   - Click **Start** to begin.
3. **To play a 3 or 4-player game**:
   - Use the **Add Player** button to configure 3 or 4 slots.
   - Fill the bot slots with **Catanatron** or **Weighted Random** bots.
   - *Note: Safety validation in the frontend will show an alert and disable the start button if the RL Bot is chosen in a 3 or 4-player game.*

---

## Method 2: Play in the Terminal (CLI)

Catanatron provides a CLI runner for interactive terminal play.

### 1. Setup Environment
Activate your Python virtual environment and ensure dependencies are installed:
```bash
source .venv/bin/activate
pip install -e ".[web,gym,dev]"
```

### 2. Play a 2-Player Game Against the RL Bot
We register the custom RL player in the CLI registry via the wrapper script `sandbox/rl_cli_play.py`. Run:
```bash
catanatron-play --code=sandbox/rl_cli_play.py --players=H,RL --num=1
```
- Enter the index of the action you want to take (e.g. `>>> 4`) when prompted in your terminal.

### 3. Play a 3-Player or 4-Player Game
Since the RL bot is limited to 2 players, use the Minimax bot (`AB`) or other bots for 3 or 4 players:
```bash
# 3-Player Game (1 Human, 1 Minimax Bot at depth 2, 1 Random Bot)
catanatron-play --players=H,AB:2,R --num=1
```
