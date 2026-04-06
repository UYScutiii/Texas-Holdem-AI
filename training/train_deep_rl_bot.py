"""
training/train_deep_rl_bot.py
─────────────────────────────
Multi-player PPO training for RLBot.

Table layout
────────────
Every episode seats the RL bot at a 4-player table alongside all three
pool opponents simultaneously:
  • CFRBot         — loads models/cfr_regret.pkl if it exists
  • MonteCarloBot  — 500 simulations per decision
  • GTOBot         — balanced mixed strategy

Seat assignments are shuffled randomly each episode so the RL bot
experiences every table position (BTN, SB, BB, UTG) equally.

Reward signal
─────────────
Per-hand normalised chip delta only:
    reward = (chips_after_RL − chips_before_RL) / max(chips_before_RL, 1)

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


# ── Opponent pool (built once before the training loop) ───────────────────────

def _build_opponent_pool() -> list[tuple[str, BotAdapter]]:
    """
    Construct every pool opponent exactly once.
    Returns a list of (name, adapter) pairs that are reused for the entire
    training run — no bot is ever reconstructed or reloaded inside the loop.
    """
    path = CFR_PROFILE_PATH if os.path.exists(CFR_PROFILE_PATH) else None
    pool = [
        ("cfr",   _PlayerViewAdapter(CFRBot(profile_path=path))),
        ("mc200", _PlayerViewAdapter(MonteCarloBot(simulations=200))),
        ("gto",   _PlayerViewAdapter(GTOBot())),
    ]
    names = ", ".join(n for n, _ in pool)
    print(f"[pool] Built {len(pool)} opponent(s): {names}")
    return pool


# ── Main training function ────────────────────────────────────────────────────

def train_deep_rl_bot(
    num_episodes:     int = 20_000,
    chips_per_player: int = 500,
    csv_path:         str | None = None,
    lr_step_episodes: int = 20_000,
    load_checkpoint:  bool = True,
):
    """
    Train RLBot at a 4-player table alongside CFRBot, MC500, and GTOBot.

    Each episode randomly shuffles the four seats so the RL bot trains from
    every table position (BTN, SB, BB, UTG) across the run.  Reward is the
    per-hand normalised chip delta for the RL seat only.
    """
    print("=" * 70)
    print("TRAINING RLBot  (4-player multi-opponent table)")
    print("=" * 70)
    print(f"Episodes:            {num_episodes}")
    print(f"Chips per player:    {chips_per_player}")
    print(f"Hidden size:         {HIDDEN_SIZE}")
    print(f"Opponent pool:       cfr, mc200, gto  (seats shuffled each ep)")
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
            "episode", "rl_seat", "won", "hands_played",
            "episode_reward", "rolling_wr", "avg_reward", "lr",
        ])

    # ── Build opponent pool once (no reloading inside the loop) ──────────────
    opponent_pool = _build_opponent_pool()   # list of (name, BotAdapter)

    # ── Training state ────────────────────────────────────────────────────────
    table          = Table()
    wins           = 0
    recent_rewards = deque(maxlen=100)

    print(f"[train] Starting training loop …\n")

    # ── Main training loop ────────────────────────────────────────────────────
    for episode in range(1, num_episodes + 1):

        # Flush the previous episode's trajectory into the PPO batch buffer
        rl_bot.end_episode()
        rl_bot.opponent_stats = {}

        # ── LR decay ─────────────────────────────────────────────────────────
        if episode > 1 and (episode - 1) % lr_step_episodes == 0:
            num_decays = (episode - 1) // lr_step_episodes
            new_lr = INITIAL_LR * (LR_DECAY_FACTOR ** num_decays)
            for pg in rl_bot.optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"  [LR] Decayed to {new_lr:.2e} at episode {episode}")

        # ── Build shuffled 4-player roster ───────────────────────────────────
        # Combine all 3 pool opponents with the RL bot, shuffle, then assign
        # player IDs P1–P4 in order.  The RL bot ends up at a random seat
        # (and therefore a random table position) each episode.
        roster = list(opponent_pool) + [("rl", None)]   # None → use rl_bot
        random.shuffle(roster)

        seats: list[Seat]          = []
        bots:  dict[str, BotAdapter] = {}
        rl_pid: str                = ""

        for i, (name, adapter) in enumerate(roster):
            pid = f"P{i + 1}"
            seats.append(Seat(player_id=pid, chips=chips_per_player))
            if name == "rl":
                bots[pid] = InProcessBot(rl_bot)
                rl_pid    = pid
            else:
                bots[pid] = InProcessBot(adapter)

        rl_seat_label = rl_pid   # e.g. "P3" — logged in CSV / progress

        # ── Play until one player remains or hand-count safety limit ──────────
        hand_count     = 0
        dealer_index   = 0
        episode_reward = 0.0
        winner_pid     = None

        while True:
            active_seats = [s for s in seats if s.chips > 0]

            if len(active_seats) <= 1:
                winner_pid = active_seats[0].player_id if active_seats else None
                break

            # Capture RL bot's chip count before this hand (0 if already bust)
            rl_seat_before = next(
                (s for s in active_seats if s.player_id == rl_pid), None
            )
            chips_before_rl = rl_seat_before.chips if rl_seat_before else 0

            result = table.play_hand(
                seats=active_seats,
                small_blind=1,
                big_blind=2,
                dealer_index=dealer_index % len(active_seats),
                bot_for={s.player_id: bots[s.player_id] for s in active_seats},
                on_event=None,
                log_decisions=False,
            )

            # Per-hand reward: normalised chip delta for the RL bot only.
            # Seat objects are mutated in-place by the engine, so reading
            # chips from the Seat after play_hand gives the post-hand count.
            if rl_pid in result and chips_before_rl > 0:
                rl_seat_after  = next(
                    (s for s in seats if s.player_id == rl_pid), None
                )
                chips_after_rl = rl_seat_after.chips if rl_seat_after else 0
                hand_reward    = (chips_after_rl - chips_before_rl) / chips_before_rl
                rl_bot.record_reward(hand_reward)
                episode_reward += hand_reward

            dealer_index = (dealer_index + 1) % len(seats)
            hand_count  += 1

            if hand_count > 10_000:          # safety limit
                winner_pid = max(seats, key=lambda s: s.chips).player_id
                break

        # ── Episode outcome ───────────────────────────────────────────────────
        won = (winner_pid == rl_pid)
        if won:
            wins += 1
        recent_rewards.append(episode_reward)

        # ── CSV row ───────────────────────────────────────────────────────────
        if csv_writer:
            rolling_wr = wins / episode
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            current_lr = rl_bot.optimizer.param_groups[0]["lr"]
            csv_writer.writerow([
                episode, rl_seat_label, int(won), hand_count,
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
                f"rl_seat={rl_seat_label}"
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
    print(f"  Wins (1st place):      {wins} / {num_episodes}  ({final_wr:.1%})")
    print(f"  Avg reward (last 100): {avg_final:+.3f}")
    print(f"{'=' * 70}")

    return rl_bot


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train RLBot at a 4-player table alongside "
            "CFRBot, MonteCarloBot(200), and GTOBot using PPO."
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
