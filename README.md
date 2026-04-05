# Texas Hold'em Bot

A poker engine with pluggable AI bots. Comes with five bot types ranging from simple heuristics to neural network-based and game-theoretic strategies, a live tournament UI, and training pipelines for the ML and RL bots.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch, Matplotlib, treys.

## Running

**Tournament UI** (live chart, click Play to start):
```bash
python run_tournament.py
```

**Single tournament** (runs to completion, saves a chart to `output/`):
```bash
python run_local_match.py
```

**Batch statistics** (run many tournaments, report win rates):
```bash
python run_tournament_stats.py --tournaments 100 --chips 500
```

**Testing a specific RL model checkpoint**:
```bash
# Test a specific RL model in a single tournament
python run_local_match.py --rl_model models/rl_model_run3.pt

# Test a specific RL model in batch statistics
python run_tournament_stats.py --tournaments 50 --rl_model models/rl_model_run3.pt

# Test a specific RL model in the interactive UI
python run_tournament.py --rl_model models/rl_model_run3.pt
```

The `--rl_model` flag automatically rewrites any `rl` entries in the `--players` spec to use the specified model path.

## Project Structure

```
.
├── core/               Game engine, bot interface, decision logger
├── bots/               Bot implementations (Monte Carlo, Poker Mind, ML, RL, CFR)
│   └── cfr_bot.py          Monte Carlo CFR bot
├── models/             Neural network architecture and saved model weights (.pt)
├── training/           Training scripts for ML and RL bots
│   ├── train_rl_bot.py         Original fixed-opponent curriculum (random → heuristic → MC)
│   ├── train_rl_bot_mixed.py   Mixed opponent curriculum (weighted heuristic/MC pool)
│   ├── train_rl_bot_selfplay.py Self-play curriculum (random → heuristic → self-play)
│   └── train_cfr_bot.py        Pure self-play CFR training (MCCFR, Nash convergence)
├── data/               Training datasets
├── logs/               Auto-generated decision logs (JSONL)
├── output/             Tournament charts and visualizations
├── run_tournament.py        Live tournament UI (matplotlib)
├── run_local_match.py       Single tournament runner
└── run_tournament_stats.py  Batch tournament statistics
```

### core/

The game engine (`engine.py`) handles the full hand lifecycle: blinds, betting rounds, street transitions, showdowns, and pot distribution. Includes a pure-Python hand evaluator. `bot_api.py` defines the `Action`, `PlayerView`, and `BotAdapter` interfaces that all bots implement. `logger.py` writes per-decision JSONL logs used for ML training.

### bots/

All four bot implementations live here. Each bot implements an `act(state) -> Action` method. The runner scripts import directly from this folder.

### models/

Contains the `PokerMLP` network definition (`poker_mlp.py`) and any saved model weights (`ml_model.pt`, `rl_model.pt`) produced by training.

### training/

Scripts to train the ML, RL, and CFR bots. All scripts add the project root to `sys.path` so they can be run from anywhere. Three separate scripts cover different RL training strategies and one dedicated script trains the CFR bot — see the **Training Scripts** section below for guidance on which to use.

### logs/

Decision logs generated during games. Each session creates a timestamped `.jsonl` file with every bot decision (hole cards, board, pot, action chosen, legal actions). These feed directly into ML training.

## Bots

### Monte Carlo Bot

The strongest bot. Runs Monte Carlo simulations (default 200) to estimate equity against random opponent hands, then compares that equity to pot odds. Adjusts aggression thresholds by table position -- tighter early, looser on the button. No learning required, just brute-force probability.

### Poker Mind Bot (SmartBot)

A heuristic bot that doesn't simulate anything. Preflop, it classifies hands into tiers (premium pairs, broadway cards, trash) and adjusts for position. Postflop, it uses the hand evaluator to estimate strength on a 0-1 scale and plays accordingly: bet strong hands, check/call medium ones, fold weak ones. Has a small bluff frequency built in.

### ML Bot

Supervised learning bot using a small feedforward network (PokerMLP, 26 input features, 128 hidden units, 6 output classes). Trained on decision logs from other bots -- it learns to imitate their play. Features include hand strength, pot odds, position, and opponent memory (aggression/tightness/VPIP tracked during the session). Falls back to a hand-strength heuristic when the model is untrained or confidence is low.

Train it:
```bash
python training/train_ml_bot.py --log_dir logs --epochs 8

# Learn from a specific bot's decisions
python training/train_ml_bot.py --log_dir logs --filter_players P3

# Only train on winning hands
python training/train_ml_bot.py --log_dir logs --filter_winners
```

### RL Bot

Reinforcement learning bot using Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE-lambda). Learns through trial and error by playing thousands of games. Uses a deeper network (512 hidden units, dropout) with the same 26-feature input as the ML bot, plus a separate value network for the critic. Rewards are normalized chip deltas for proportional credit assignment. Supports three different training modes via three separate scripts — see **Training Scripts** below.

Train it:
```bash
# Original fixed-opponent curriculum (random -> heuristic -> Monte Carlo)
python training/train_rl_bot.py --episodes 50000 --curriculum

# Mixed opponent curriculum (weighted heuristic/MC pool, no demotion)
python training/train_rl_bot_mixed.py --episodes 50000

# Self-play curriculum (random -> heuristic -> self-play snapshots)
python training/train_rl_bot_selfplay.py --episodes 50000
```

All three scripts save models to `models/`.

### CFR Bot

Game-theoretic bot using Monte Carlo Counterfactual Regret Minimization (MCCFR). Rather than learning from trial and error, it iteratively reduces regret across sampled game trajectories until its strategy converges toward a Nash equilibrium. Unlike the RL bot, it does not use a neural network — instead it maintains a persistent regret table that updates across hands within a session.

Key design details:
- **Bet abstraction**: actions are bucketed into 33% pot, 67% pot, pot, and all-in bets, keeping the information state space tractable.
- **Card abstraction**: hole cards and board texture are mapped to hand-strength buckets rather than exact ranks/suits.
- **Regret table**: stored in memory and optionally persisted to disk between sessions so the strategy improves over multiple runs.
- Converges toward Nash equilibrium over time — the more iterations, the closer to optimal play.

Save and load the regret table:
```python
from bots.cfr_bot import CFRBot

bot = CFRBot(iterations=1000)
bot.save("models/cfr_regret.pkl")   # persist regret table

bot2 = CFRBot()
bot2.load("models/cfr_regret.pkl")  # resume from saved state
```

## Training Scripts

Three scripts train the RL bot with different opponent curricula. All share the same PPO update loop, GAE-lambda advantage estimation, logging, and CLI arguments.

### train_rl_bot.py
Fixed-opponent curriculum. Two modes:
- **Without `--curriculum`**: trains against a fixed opponent (default: montecarlo). Fast for `heuristic` or `self`, but **very slow against montecarlo** (200 simulations per decision).
- **With `--curriculum`**: walks through **random → heuristic → montecarlo → self-play**. Note that the montecarlo stage is **extremely slow in practice** due to simulation cost per decision. Recommended only if you have significant compute time.

### train_rl_bot_mixed.py
Mixed opponent curriculum (weighted heuristic/MC pool). Smoother curriculum with no demotion as win rate improves.

### train_rl_bot_selfplay.py
**Recommended training script.** Three-stage curriculum: **random → heuristic → self-play** (skips Monte Carlo entirely for speed). Loads from `models/rl_model_run2.pt` if available, saves final model to `models/rl_model_run3.pt`.

**Typical progression**: train with `train_rl_bot_selfplay.py` for the fastest results, or start with `train_rl_bot.py` without `--curriculum` for a stable baseline, then continue with `train_rl_bot_mixed.py` once the bot can beat Monte Carlo.

### train_cfr_bot.py
Dedicated self-play training script for the CFR bot. A single CFRBot instance plays both seats simultaneously, building a shared regret table from both perspectives on every hand — the theoretically correct approach for Nash convergence. Saves the regret table to `models/cfr_regret.pkl` periodically and resumes automatically from that file if it exists on startup.

```bash
# Default: 50,000 episodes at 200 MCCFR rollouts per decision
python training/train_cfr_bot.py

# Higher quality (slower)
python training/train_cfr_bot.py --tournaments 100000 --iterations 500

# Resume from a previous run or save to a custom path
python training/train_cfr_bot.py --profile models/cfr_v2.pkl
```

## Adding a Bot

Create a file in `bots/` and implement `act()`:

```python
from core.bot_api import Action, PlayerView

class MyBot:
    def act(self, state: PlayerView) -> Action:
        # state has: hole_cards, board, pot, to_call, legal_actions, stacks, position, etc.
        return Action("call")
```

Then add it to whichever runner script you want to use.
