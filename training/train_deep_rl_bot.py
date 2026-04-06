"""
training/train_deep_rl_bot.py
─────────────────────────────
Multi-opponent PPO training for RLBot.

Opponent pool
─────────────
Each episode, one opponent is drawn *uniformly at random* (without repeating
the same one two episodes in a row) from:
  • CFRBot         — loads models/cfr_regret.pkl if it exists
  • MonteCarloBot  — 500 simulations per decision
  • GTOBot         — balanced mixed strategy

Reward signal
─────────────
Per-hand normalised chip delta only:
    reward = (chips_after − chips_before) / max(chips_before, 1)

No asymmetric terminal win/loss bonus is applied.

Checkpoint / output
───────────────────
  Loads  models/deep_rl_model.pt  if it exists, otherwise starts fresh.
  Saves  models/deep_rl_model.pt  at the end of training.
  CSV    output/rl_training_log_deep.csv

Usage
─────
    python training/train_deep_rl_bot.py [--episodes N] [--chips N]
                                          [--csv PATH] [--lr_step N]
                                          [--no_load]
"""

import os
import sys
import csv
import random
import argparse
from collections import deque

# ── Project-root import fix ───────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from core.engine import Table, Seat, InProcessBot
from core.bot_api import BotAdapter, PlayerView, Action
from bots.rl_bot import RLBot
from bots.cfr_bot import CFRBot
from bots.monte_carlo_bot import MonteCarloBot
from bots.gto_bot import GTOBot


# ── Constants ─────────────────────────────────────────────────────────────────

FINAL_MODEL_PATH = "models/deep_rl_model.pt"
DEFAULT_CSV_PATH  = "output/rl_training_log_deep.csv"
CFR_PROFILE_PATH  = "models/cfr_regret.pkl"
HIDDEN_SIZE       = 512
INITIAL_LR        = 3e-4
LR_DECAY_FACTOR   = 0.5


# ── Minimal BotAdapter wrapper ────────────────────────────────────────────────

class _PlayerViewAdapter(BotAdapter):
    """Thin wrapper so any bot with .act(PlayerView) fits the engine interface."""
    def __init__(self, bot):
        self.bot = bot

    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)


# ── Opponent factory helpers ──────────────────────────────────────────────────

def _make_cfr() -> BotAdapter:
    """Construct CFRBot; silently skips profile load if file is absent."""
    path = CFR_PROFILE_PATH if os.path.exists(CFR_PROFILE_PATH) else None
    return _PlayerViewAdapter(CFRBot(profile_path=path))


def _make_mc500() -> BotAdapter:
    return _PlayerViewAdapter(MonteCarloBot(simulations=500))


def _make_gto() -> BotAdapter:
    return _PlayerViewAdapter(GTOBot())


# Pool entry format: {"name": str, "factory": callable → BotAdapter}
OPPONENT_POOL = [
    {"name": "cfr",   "factory": _make_cfr},
    {"name": "mc500", "factory": _make_mc500},
    {"name": "gto",   "factory": _make_gto},
]


def _sample_opponent(prev_idx: int | None) -> tuple[int, BotAdapter]:
    """
    Pick a random pool index, avoiding the same opponent as last episode.
    Returns (new_idx, fresh_bot_instance).
    """
    n = len(OPPONENT_POOL)
    choices = [i for i in range(n) if i != prev_idx]
    idx = random.choice(choices)
    bot = OPPONENT_POOL[idx]["factory"]()
    return idx, bot


# ── Main training function ────────────────────────────────────────────────────

def train_deep_rl_bot(
    num_episodes:    int = 20_000,
    chips_per_player: int = 500,
    csv_path:        str | None = None,
    lr_step_episodes: int = 20_000,
    load_checkpoint: bool = True,
):
    """
    Train RLBot against a rotating opponent pool with pure per-hand rewards.

    Args:
        num_episodes:     Number of tournament episodes (each: play until one
                          player is eliminated, up to 10 000 hands).
        chips_per_player: Starting chip count for both players.
        csv_path:         Optional path for the per-episode CSV log.
        lr_step_episodes: Halve the learning rate every this many episodes.
        load_checkpoint:  If True, load FINAL_MODEL_PATH when it exists.
    """
    print("=" * 70)
    print("TRAINING RLBot  (deep multi-opponent pool)")
    print("=" * 70)
    print(f"Episodes:            {num_episodes}")
    print(f"Chips per player:    {chips_per_player}")
    print(f"Hidden size:         {HIDDEN_SIZE}")
    print(f"Opponent pool:       {', '.join(e['name'] for e in OPPONENT_POOL)}")
    print(f"Reward signal:       per-hand normalised chip delta (no terminal bonus)")
    print(f"Checkpoint:          {FINAL_MODEL_PATH}")
    print(f"LR step every:       {lr_step_episodes} episodes")
    print("=" * 70)
    print()

    # ── Build RLBot ───────────────────────────────────────────────────────────
    rl_bot = RLBot(
        model_path="",          # skip internal auto-load
        training_mode=True,
        learning_rate=INITIAL_LR,
        starting_chips=chips_per_player,
    )

    # Graceful checkpoint load
    if load_checkpoint and os.path.exists(FINAL_MODEL_PATH):
        try:
            ckpt = torch.load(FINAL_MODEL_PATH, map_location=rl_bot.device)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                rl_bot.policy_net.load_state_dict(ckpt["policy"])
                rl_bot.value_net.load_state_dict(ckpt["value"])
            else:
                rl_bot.policy_net.load_state_dict(ckpt)
            rl_bot.policy_net.train()
            rl_bot.value_net.train()
            print(f"[checkpoint] Loaded from {FINAL_MODEL_PATH}")
        except RuntimeError as e:
            print(f"[checkpoint] Size mismatch loading {FINAL_MODEL_PATH} "
                  f"— starting fresh.\n  Detail: {e}")
        except Exception as e:
            print(f"[checkpoint] Could not load {FINAL_MODEL_PATH}: {e} "
                  f"— starting fresh")
    else:
        reason = "disabled" if not load_checkpoint else "not found"
        print(f"[checkpoint] {FINAL_MODEL_PATH} {reason} — starting fresh")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    csv_file   = None
    csv_writer = None
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        csv_file   = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "episode", "opponent", "won", "hands_played",
            "episode_reward", "rolling_wr", "avg_reward", "lr",
        ])

    # ── Training state ────────────────────────────────────────────────────────
    table          = Table()
    wins           = 0
    recent_rewards = deque(maxlen=100)
    prev_opp_idx: int | None = None

    print(f"[train] Starting training loop …\n")

    # ── Main training loop ────────────────────────────────────────────────────
    for episode in range(1, num_episodes + 1):

        # Flush previous episode's buffer into the batch
        rl_bot.end_episode()
        rl_bot.opponent_stats = {}

        # ── LR decay ─────────────────────────────────────────────────────────
        if episode > 1 and (episode - 1) % lr_step_episodes == 0:
            num_decays = (episode - 1) // lr_step_episodes
            new_lr = INITIAL_LR * (LR_DECAY_FACTOR ** num_decays)
            for pg in rl_bot.optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"  [LR] Decayed to {new_lr:.2e} at episode {episode}")

        # ── Sample opponent (no repeat) ───────────────────────────────────────
        opp_idx, opponent_bot = _sample_opponent(prev_opp_idx)
        prev_opp_idx          = opp_idx
        opp_name              = OPPONENT_POOL[opp_idx]["name"]

        # ── Set up table seats ────────────────────────────────────────────────
        seats = [
            Seat(player_id="P1", chips=chips_per_player),  # opponent
            Seat(player_id="P2", chips=chips_per_player),  # RL agent
        ]
        bots = {
            "P1": InProcessBot(opponent_bot),
            "P2": InProcessBot(rl_bot),
        }

        # ── Play until elimination or hand-count safety limit ─────────────────
        hand_count    = 0
        dealer_index  = 0
        episode_reward = 0.0

        while True:
            active_seats = [s for s in seats if s.chips > 0]
            if len(active_seats) <= 1:
                winner = active_seats[0].player_id if active_seats else None
                break

            chips_before_p2 = sum(s.chips for s in seats if s.player_id == "P2")

            result = table.play_hand(
                seats=active_seats,
                small_blind=1,
                big_blind=2,
                dealer_index=dealer_index % len(active_seats),
                bot_for={s.player_id: bots[s.player_id] for s in active_seats},
                on_event=None,
                log_decisions=False,
            )

            # Per-hand reward: normalised chip delta only (no terminal bonus)
            chips_after_p2 = sum(s.chips for s in seats if s.player_id == "P2")
            if "P2" in result:
                hand_reward = (chips_after_p2 - chips_before_p2) / max(chips_before_p2, 1)
                rl_bot.record_reward(hand_reward)
                episode_reward += hand_reward

            dealer_index = (dealer_index + 1) % len(seats)
            hand_count  += 1

            if hand_count > 10_000:          # safety limit
                winner = max(seats, key=lambda s: s.chips).player_id
                break

        # ── Episode outcome ───────────────────────────────────────────────────
        won = (winner == "P2")
        if won:
            wins += 1
        recent_rewards.append(episode_reward)

        # ── CSV row ───────────────────────────────────────────────────────────
        if csv_writer:
            rolling_wr = wins / episode
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            current_lr = rl_bot.optimizer.param_groups[0]["lr"]
            csv_writer.writerow([
                episode, opp_name, int(won), hand_count,
                f"{episode_reward:.4f}", f"{rolling_wr:.4f}",
                f"{avg_reward:.4f}", f"{current_lr:.2e}",
            ])

        # ── Progress log every 100 episodes ──────────────────────────────────
        if episode % 100 == 0:
            rolling_wr = wins / episode
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            current_lr = rl_bot.optimizer.param_groups[0]["lr"]
            print(
                f"  ep={episode:>6}  wins={wins:>5}  wr={rolling_wr:.1%}  "
                f"avg_r={avg_reward:+.3f}  lr={current_lr:.1e}  "
                f"last_opp={opp_name}"
            )

    # ── End of training ───────────────────────────────────────────────────────
    rl_bot.flush_buffer()

    os.makedirs("models", exist_ok=True)
    rl_bot.save_model(FINAL_MODEL_PATH)
    print(f"\nModel saved to {FINAL_MODEL_PATH}")

    if csv_file:
        csv_file.close()
        print(f"Training log saved to {csv_path}")

    final_wr  = wins / num_episodes if num_episodes > 0 else 0.0
    avg_final = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
    print(f"\n{'=' * 70}")
    print("Training complete.")
    print(f"  Episodes:              {num_episodes}")
    print(f"  Wins:                  {wins} / {num_episodes}  ({final_wr:.1%})")
    print(f"  Avg reward (last 100): {avg_final:+.3f}")
    print(f"{'=' * 70}")

    return rl_bot


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train RLBot against a rotating pool of CFRBot, "
            "MonteCarloBot(500), and GTOBot using PPO."
        )
    )
    parser.add_argument(
        "--episodes", type=int, default=20_000,
        help="Number of tournament episodes (default: 20000)",
    )
    parser.add_argument(
        "--chips", type=int, default=500,
        help="Starting chips per player (default: 500)",
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV_PATH,
        help=f"Path for per-episode CSV log (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--lr_step", type=int, default=20_000,
        help="Halve LR every this many episodes (default: 20000)",
    )
    parser.add_argument(
        "--no_load", action="store_true",
        help="Ignore any existing checkpoint and start fresh",
    )
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    train_deep_rl_bot(
        num_episodes=args.episodes,
        chips_per_player=args.chips,
        csv_path=args.csv,
        lr_step_episodes=args.lr_step,
        load_checkpoint=not args.no_load,
    )
