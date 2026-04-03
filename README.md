# Texas Hold'em Bot

A poker engine with pluggable AI bots. Comes with four bot types ranging from simple heuristics to neural network-based strategies, a live tournament UI, and training pipelines for the ML and RL bots.

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

## Project Structure

```
.
├── core/               Game engine, bot interface, decision logger
├── bots/               Bot implementations (Monte Carlo, Poker Mind, ML, RL)
├── models/             Neural network architecture and saved model weights (.pt)
├── training/           Training scripts for ML and RL bots
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

Scripts to train the ML and RL bots. Both add the project root to `sys.path` so they can be run from anywhere.

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

Reinforcement learning bot using the REINFORCE policy gradient algorithm. Learns through trial and error by playing thousands of games. Uses a deeper network (512 hidden units, dropout) with the same 26-feature input as the ML bot. Supports curriculum training that starts against weak opponents (random) and promotes to stronger ones (heuristic, then Monte Carlo) once win rate crosses a threshold.

Train it:
```bash
# Against Monte Carlo (recommended)
python training/train_rl_bot.py --episodes 50000 --opponent montecarlo

# Curriculum mode: random -> heuristic -> Monte Carlo
python training/train_rl_bot.py --episodes 50000 --curriculum

# Self-play
python training/train_rl_bot.py --episodes 50000 --opponent self
```

Both training scripts save models to `models/`.

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
