# Texas Hold'em Bot

A poker engine with pluggable AI bots. Ships with nine bot types ranging from simple heuristics to neural-network, reinforcement-learning, game-theoretic, and opponent-modeling strategies, plus a live tournament UI, batch statistics runner, and training pipelines for the ML, RL, and CFR bots.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch (>= 1.9), Matplotlib (>= 3.5), treys (>= 0.1.8).

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
python run_local_match.py --rl_model models/rl_model_run3.pt
python run_tournament_stats.py --tournaments 50 --rl_model models/rl_model_run3.pt
python run_tournament.py --rl_model models/rl_model_run3.pt
```

The `--rl_model` flag automatically rewrites any `rl` entries in the `--players` spec to use the specified model path.

## Project Structure

```
.
├── core/                          Game engine, bot interface, decision logger
│   ├── engine.py                  Full hand lifecycle, hand evaluator, pot distribution
│   ├── bot_api.py                 Action, PlayerView, BotAdapter interfaces
│   └── logger.py                  Per-decision JSONL logger for ML training
│
├── bots/                          Nine bot implementations + factory
│   ├── __init__.py                Bot factory (create_bot, parse_players, escalate_blinds)
│   ├── monte_carlo_bot.py         Monte Carlo rollout equity estimation
│   ├── poker_mind_bot.py          Heuristic hand-tier classification (SmartBot)
│   ├── ml_bot.py                  Supervised learning (26-feature MLP)
│   ├── rl_bot.py                  PPO with GAE-lambda and value network
│   ├── cfr_bot.py                 Monte Carlo CFR (MCCFR, regret matching)
│   ├── icm_bot.py                 Tournament equity (Independent Chip Model)
│   ├── exploitative_bot.py        Opponent-tracking exploitation
│   ├── gto_bot.py                 GTO approximation with balanced mixed strategies
│   └── opponent_model_bot.py      Bayesian hand-range modeling
│
├── models/                        Network definitions and saved weights
│   ├── poker_mlp.py               PokerMLP architecture (3-layer feedforward)
│   ├── five_card_table.pkl        Precomputed hand evaluator lookup (~45 MB)
│   ├── cfr_regret.pkl             Heads-up CFR regret table (~13 MB)
│   ├── cfr_regret_deep.pkl        Multiway/deep-stack CFR regret table (~13 MB)
│   └── cfr_regret_pre_migration.pkl  Legacy backup (~13 MB)
│
├── training/                      Training scripts for ML, RL, and CFR bots
│   ├── train_ml_bot.py            Supervised learning on decision logs
│   ├── train_rl_bot.py            Fixed-opponent curriculum (random -> heuristic -> MC -> self-play)
│   ├── train_rl_bot_mixed.py      Mixed opponent curriculum (weighted pool, no demotion)
│   ├── train_rl_bot_selfplay.py   Self-play curriculum (random -> heuristic -> self-play)
│   ├── train_deep_rl_bot.py       4-player table training (CFR + MC + GTO opponents)
│   ├── train_cfr_bot.py           Heads-up self-play CFR training
│   ├── train_cfr_finetune.py      CFR fine-tuning against real opponent bots
│   └── train_cfr_bot_multiway.py  4-player deep-stack CFR training
│
├── logs/                          Auto-generated JSONL decision logs
├── output/                        Tournament charts (.png) and stats (.csv)
├── data/                          Training datasets (currently empty)
│
├── run_tournament.py              Live tournament UI (matplotlib, Play button)
├── run_local_match.py             Single tournament runner with chart output
├── run_tournament_stats.py        Batch statistics with multiprocessing
└── requirements.txt               Python dependencies
```

## Core Engine

The game engine (`core/engine.py`) handles the full hand lifecycle: blinds, betting rounds (preflop through river), street transitions, showdowns, side pots, and pot distribution. It includes a pure-Python hand evaluator backed by a precomputed lookup table covering all 2,598,960 five-card combinations (~45 MB, built once and cached to `models/five_card_table.pkl`).

`core/bot_api.py` defines the three interfaces all bots implement: `Action` (type + optional amount), `PlayerView` (read-only game state without opponent hole cards), and `BotAdapter` (requires `act(state) -> Action`).

`core/logger.py` writes per-decision JSONL logs used for ML training. Each entry captures hand ID, player, hole cards, board, pot, action chosen, and legal actions.

## Player Specs

Bots are created via string keys passed to `create_bot()` or as comma-separated specs to `parse_players()`:

| Key | Bot | Notes |
|-----|-----|-------|
| `mc`, `mc<N>` | MonteCarloBot | Optional sim count: `mc200`, `mc500` (default 200) |
| `smart` | SmartBot | Also accepts `smartbot`, `heuristic` |
| `ml` | MLBot | Also accepts `mlbot` |
| `rl`, `rl:<path>` | RLBot | Optional model path: `rl:models/custom.pt` |
| `cfr` | CFRBot | Loads `models/cfr_regret_deep.pkl` in inference mode |
| `icm` | ICMBot | Also accepts `icmbot` |
| `exploitative` | ExploitativeBot | Also accepts `exploitativebot` |
| `gto` | GTOBot | Also accepts `gtobot` |
| `opponentmodel` | OpponentModelBot | Also accepts `opponentmodelbot` |
| `random` | RandomBot | Uniform random legal actions |

Example: `--players mc200,smart,rl,cfr` creates a 4-player table with auto-assigned IDs (P1-P4). Named seats: `--players P1=mc200,P2=smart`.

## Bots

### Monte Carlo Bot

Runs Monte Carlo simulations (default 200) to estimate equity against random opponent hands, then compares equity to pot odds. Adjusts aggression thresholds by table position (tighter early, looser on the button). No learning required, just brute-force probability. The strongest pure heuristic bot.

### SmartBot (Poker Mind Bot)

A heuristic bot that classifies hands into tiers (premium pairs, broadway cards, trash) preflop and uses the hand evaluator to estimate strength on a 0-1 scale postflop. Plays accordingly: bet strong hands, check/call medium ones, fold weak ones. Has a small bluff frequency (~7%) built in. Fast baseline reference.

### ML Bot

Supervised learning bot using a 3-layer feedforward network (PokerMLP: 26 input features, 128 hidden units, 6 output action classes). Trained on decision logs from other bots. Features include hand strength, pot odds, position, and opponent memory (aggression, tightness, VPIP tracked from the last 10 observed actions per opponent). Falls back to a hand-strength heuristic when the model is untrained or confidence is low.

```bash
python training/train_ml_bot.py --log_dir logs --epochs 8
python training/train_ml_bot.py --log_dir logs --filter_players P3
python training/train_ml_bot.py --log_dir logs --filter_winners
```

### RL Bot

Reinforcement learning bot using Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE-lambda). Uses a 512-unit policy network with dropout and a separate 512-unit value network (critic). Same 26-feature input as the ML bot. Rewards are normalized chip deltas for proportional credit assignment, with terminal bonuses for wins/losses. Exploration rate is fixed at 10% during training. Supports four training modes via separate scripts (see Training Scripts below).

```bash
python training/train_rl_bot.py --episodes 50000 --curriculum
python training/train_rl_bot_mixed.py --episodes 50000
python training/train_rl_bot_selfplay.py --episodes 50000
python training/train_deep_rl_bot.py --episodes 50000
```

### CFR Bot

Game-theoretic bot using Monte Carlo Counterfactual Regret Minimization (MCCFR). Iteratively reduces regret across sampled game trajectories until its strategy converges toward a Nash equilibrium. Maintains a persistent regret table (not a neural network) that updates across hands.

Key design details:
- **Card abstraction**: 10 preflop buckets (hand-strength tiers) and 10 postflop buckets (Monte Carlo equity percentiles from 20 rollouts).
- **Bet abstraction**: 6 abstract actions (fold, check/call, 33% pot, 67% pot, pot, all-in).
- **Action history**: compressed into 8-character tokens (F/K/C/S/M/P/A) for information-set keys.
- **Regret table**: persisted to disk between sessions so the strategy improves over multiple runs.
- **Inference mode**: skips online regret updates to avoid corrupting loaded strategies during play.

Three saved profiles exist: `cfr_regret.pkl` (heads-up), `cfr_regret_deep.pkl` (multiway/deep-stack, used by default at inference), and `cfr_regret_pre_migration.pkl` (legacy backup).

### ICM Bot

Tournament equity-aware bot using Malmuth-Harville Independent Chip Model (ICM) calculations. Converts chip stacks into tournament equity (prize shares) and makes decisions that maximize equity preservation rather than raw chip EV. Plays aggressively with a large stack and tightens up when its own stack is at risk.

### Exploitative Bot

Adapts mid-session by tracking per-opponent statistics: VPIP, aggression factor (AF), and fold-to-aggression rate (FTA). Falls back to tight-aggressive (TAG) defaults until it has 5+ hands of history on an opponent, then exploits detected tendencies: bluffs against high-FTA players, value-bets against calling stations, and traps against hyper-aggressors.

### GTO Bot

Approximates Game Theory Optimal play using position-aware preflop hand-range charts (early, mid, late, blinds) and balanced mixed strategies postflop. Targets a 2:1 value-to-bluff ratio on the river. Continuation-bet frequency (60-70%), check-raise frequency (12-18%), and probe bets are all tuned for balance. Non-deterministic by design.

### Opponent Model Bot

Bayesian hand-range modeling. Maintains a probability distribution over five hand-strength buckets (trash, weak, medium, strong, premium) per opponent and updates via likelihood multipliers from observed actions. Runs Monte Carlo equity against the weighted opponent range rather than random hands for more accurate pot-odds calculations as the hand progresses.

## Training Scripts

### RL Training

Four scripts train the RL bot with different strategies. All share the same PPO update loop, GAE-lambda, and CLI arguments.

**train_rl_bot.py** -- Fixed-opponent curriculum with optional promotion/demotion. Walks through random, heuristic, Monte Carlo, and self-play stages. The Monte Carlo stage is very slow (200 sims per opponent decision). Supports `--curriculum` flag.

**train_rl_bot_mixed.py** -- Mixed opponent curriculum with continuous weight shifts between heuristic and Monte Carlo opponents (initial 80/20, shifting 5% per threshold crossing). Includes survival bonuses and early-elimination penalties in reward shaping. No demotion. Transitions to self-play when MC weight reaches 80%.

**train_rl_bot_selfplay.py** -- **Recommended for fastest results.** Three-stage curriculum: random, heuristic, self-play (skips Monte Carlo entirely). Gracefully handles checkpoint loading across architecture changes (256 to 512 hidden units). Saves snapshots every 500 episodes during self-play. Loads from `models/rl_model_run2.pt`, saves to `models/rl_model_run3.pt`.

**train_deep_rl_bot.py** -- 4-player table training. Pits the RL bot against CFR, Monte Carlo (200 sims), and GTO opponents simultaneously with random seat assignment each episode. No curriculum stages. Faster LR decay (every 20K episodes vs 30K). Best used after the bot has a solid foundation from one of the other scripts.

### CFR Training

**train_cfr_bot.py** -- Heads-up self-play. A single CFRBot plays both seats, building a shared regret table from both perspectives (theoretically correct for Nash convergence). Expected outcome: ~50% win rate. Saves to `models/cfr_regret.pkl`.

```bash
python training/train_cfr_bot.py
python training/train_cfr_bot.py --tournaments 100000 --iterations 500
python training/train_cfr_bot.py --profile models/cfr_v2.pkl
```

**train_cfr_bot_multiway.py** -- 4-player deep-stack training (1000 chips, 5/10 blinds, 1.5x escalation every 50 hands). Four CFR instances share one regret table. Saves to `models/cfr_regret_deep.pkl`.

**train_cfr_finetune.py** -- Phase 2 fine-tuning. Loads a pre-trained profile and refines it against real bots (smart, mc100) in 4-player games. Reduced iterations (50 vs 200) for speed. Defaults to overwriting the input profile.

### ML Training

**train_ml_bot.py** -- Supervised learning on JSONL decision logs. Trains PokerMLP with Adam optimizer, ReduceLROnPlateau scheduler, 80/20 train/val split. Supports filtering by player (`--filter_players`) or winning hands only (`--filter_winners`). Requires decision logs in `logs/` (generate by running tournaments with logging enabled).

```bash
python training/train_ml_bot.py --log_dir logs --epochs 8
```

## Adding a Bot

Create a file in `bots/` and implement `act()`:

```python
from core.bot_api import Action, PlayerView

class MyBot:
    def act(self, state: PlayerView) -> Action:
        # state has: hole_cards, board, pot, to_call, legal_actions,
        #            stacks, position, history, etc.
        return Action("call")
```

Then register it in `bots/__init__.py` by adding a key-to-import mapping in `create_bot()`.

## Known Limitations

- **No web UI** -- visualization is matplotlib-only (local).
- **CFR abstraction** -- 10 postflop buckets and 20 equity rollouts are coarse; finer abstractions would improve play quality at the cost of training time and memory.
- **ML bot feature alignment** -- there are minor mismatches between feature encoding at training time and inference time (normalization, memory windowing) that can reduce model effectiveness.
- **Exploration decay** -- the RL bot's exploration rate (10%) is fixed; a decay schedule would help the bot sharpen its play in later training stages.
- **No position encoding in CFR info-sets** -- the CFR bot doesn't distinguish between positions (BTN vs BB) when building its strategy, limiting its ability to learn position-dependent play.
