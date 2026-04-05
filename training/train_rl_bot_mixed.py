"""
Train the RL bot with a mixed opponent curriculum.

Instead of discrete curriculum stages with promotion/demotion, each episode
randomly samples an opponent from a weighted pool.  The weights start at
80 % heuristic / 20 % Monte Carlo and gradually shift toward more Monte Carlo
as the rolling win rate improves.

Transition rules
────────────────
• Rolling window: 1 000 episodes.
• When rolling WR > 55 %, shift 5 % from heuristic → Monte Carlo.
• Weights are capped at 20 % heuristic / 80 % Monte Carlo.
• Weights only move forward — no demotion / backward shift.
• When MC weight reaches 80 % AND rolling WR sustains above 55 %,
  the bot transitions to pure self-play for the remainder of training.
"""
import os
import sys
import csv
import random
from collections import deque

# Add project root to path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.engine import Table, Seat, InProcessBot
from core.bot_api import BotAdapter, PlayerView, Action
from bots.rl_bot import RLBot
from bots.monte_carlo_bot import MonteCarloBot
from bots.poker_mind_bot import SmartBot
import argparse
import torch


class PlayerViewAdapter(BotAdapter):
    """Adapter for bots that expect PlayerView."""
    def __init__(self, bot):
        self.bot = bot
    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)


# ── Mixed-pool opponent factories ────────────────────────────────────────────

def _make_heuristic():
    return PlayerViewAdapter(SmartBot())

def _make_montecarlo():
    return PlayerViewAdapter(MonteCarloBot(simulations=200))


# ── Fixed-opponent helper (for --opponent CLI flag) ──────────────────────────

def _make_opponent(opponent_type):
    """Build a single opponent from a CLI flag value."""
    if opponent_type == "self":
        return InProcessBot(RLBot(training_mode=False))
    if opponent_type == "montecarlo":
        return _make_montecarlo()
    return _make_heuristic()


# ── Training loop ────────────────────────────────────────────────────────────

# Mixed-pool constants
INITIAL_HEURISTIC_WEIGHT = 0.80
INITIAL_MC_WEIGHT        = 0.20
SHIFT_STEP               = 0.05   # shift per threshold crossing
MIN_HEURISTIC_WEIGHT     = 0.20   # floor for heuristic
MAX_MC_WEIGHT            = 0.80   # ceiling for MC

ROLLING_WINDOW           = 1000   # episodes in the rolling WR window
SHIFT_THRESHOLD          = 0.55   # WR above which we shift weights
SELFPLAY_WR_THRESHOLD    = 0.55   # WR to sustain for self-play transition

SNAPSHOT_DEFAULT         = 500    # self-play snapshot interval


def train_rl_bot_mixed(num_episodes=10_000, chips_per_player=500,
                       opponent_type="montecarlo", csv_path=None,
                       lr_step_episodes=30_000, snapshot_every=SNAPSHOT_DEFAULT):
    """
    Train RL bot with a mixed opponent curriculum.

    Args:
        num_episodes: Number of tournament episodes to play.
        chips_per_player: Starting chips per player each episode.
        opponent_type: Fixed opponent when mixed curriculum is bypassed
                       (kept for argparse compatibility; ignored in mixed mode).
        csv_path: Path to write per-episode CSV log.
        lr_step_episodes: Reduce LR by 0.5x every this many episodes.
        snapshot_every: Save self-play snapshot every N episodes in self-play.
    """
    print("=" * 70)
    print("TRAINING RL BOT — MIXED OPPONENT CURRICULUM")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Chip stack per player: {chips_per_player}")
    print(f"Initial weights: heuristic={INITIAL_HEURISTIC_WEIGHT:.0%}  "
          f"montecarlo={INITIAL_MC_WEIGHT:.0%}")
    print(f"Shift step: {SHIFT_STEP:.0%} toward MC when rolling WR > "
          f"{SHIFT_THRESHOLD:.0%}")
    print(f"Self-play transition: MC weight ≥ {MAX_MC_WEIGHT:.0%} AND "
          f"rolling WR > {SELFPLAY_WR_THRESHOLD:.0%}")
    print(f"Rolling window: {ROLLING_WINDOW} episodes")
    print(f"LR step every: {lr_step_episodes} episodes")
    print(f"Loading checkpoint: models/rl_model_run1.pt")
    print("=" * 70)
    print()

    # ── Create RL bot, loading from the run-1 checkpoint ─────────────
    rl_bot = RLBot(
        training_mode=True,
        learning_rate=3e-4,
        model_path="models/rl_model_run1.pt",
    )

    # LR scheduler bookkeeping
    initial_lr = 3e-4
    lr_decay_factor = 0.5

    table = Table()

    wins = 0
    total_chips = 0
    recent_rewards = deque(maxlen=100)

    # ── Mixed-pool bookkeeping ───────────────────────────────────────
    heuristic_weight = INITIAL_HEURISTIC_WEIGHT
    mc_weight        = INITIAL_MC_WEIGHT
    rolling_results  = deque(maxlen=ROLLING_WINDOW)  # 1 = win, 0 = loss
    in_selfplay      = False

    # Track the last WR at which we shifted, to avoid shifting multiple
    # times at the same WR level within the same window.
    last_shift_wr    = 0.0

    # CSV logging
    csv_file = None
    csv_writer = None
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "episode", "won", "reward", "rolling_wr", "avg_reward",
            "lr", "stage", "heuristic_w", "mc_w",
        ])

    for episode in range(1, num_episodes + 1):
        # Reset for new tournament
        rl_bot.end_episode()
        rl_bot.opponent_stats = {}

        # ── LR scheduling ────────────────────────────────────────────
        if episode > 1 and (episode - 1) % lr_step_episodes == 0:
            num_decays = (episode - 1) // lr_step_episodes
            new_lr = initial_lr * (lr_decay_factor ** num_decays)
            for pg in rl_bot.optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"  [LR] Reduced to {new_lr:.2e} at episode {episode}")

        # ── Self-play snapshot ───────────────────────────────────────
        if in_selfplay and episode % snapshot_every == 0:
            os.makedirs("models", exist_ok=True)
            rl_bot.save_model("models/rl_selfplay_snapshot.pt")

        # ── Build opponent for this episode ──────────────────────────
        seats = [
            Seat(player_id="P1", chips=chips_per_player),
            Seat(player_id="P2", chips=chips_per_player),
        ]

        if in_selfplay:
            opponent_bot = InProcessBot(
                RLBot(training_mode=False,
                      model_path="models/rl_selfplay_snapshot.pt",
                      use_fallback=True)
            )
            stage_name = "selfplay"
        else:
            # Sample from the weighted pool
            if random.random() < mc_weight:
                opponent_bot = _make_montecarlo()
                stage_name = "montecarlo"
            else:
                opponent_bot = _make_heuristic()
                stage_name = "heuristic"

        bots = {
            "P1": opponent_bot,
            "P2": InProcessBot(rl_bot),
        }

        # ── Play tournament until winner ─────────────────────────────
        hand_count = 0
        dealer_index = 0
        initial_chips_p2 = chips_per_player

        while True:
            active_seats = [s for s in seats if s.chips > 0]
            if len(active_seats) <= 1:
                winner = active_seats[0].player_id if active_seats else None
                break

            chips_before = sum(s.chips for s in seats if s.player_id == "P2")

            result = table.play_hand(
                seats=active_seats,
                small_blind=1,
                big_blind=2,
                dealer_index=dealer_index % len(active_seats),
                bot_for={s.player_id: bots[s.player_id] for s in active_seats},
                on_event=None,
                log_decisions=False,
            )

            # Per-hand reward: normalised chip delta
            chips_after = sum(s.chips for s in seats if s.player_id == "P2")
            hand_reward = (chips_after - chips_before) / max(chips_before, 1)

            if "P2" in result:
                rl_bot.record_reward(hand_reward)

            dealer_index = (dealer_index + 1) % len(seats)
            hand_count += 1

            if hand_count > 10_000:  # safety limit
                winner = max(seats, key=lambda s: s.chips).player_id
                break

        # ── Final outcome ────────────────────────────────────────────
        final_chips_p2 = sum(s.chips for s in seats if s.player_id == "P2")
        won = winner == "P2"

        if won:
            wins += 1

        final_reward = (final_chips_p2 - initial_chips_p2) / max(initial_chips_p2, 1)

        # Terminal bonus
        final_bonus = 1.0 if won else -0.5
        rl_bot.record_reward(final_bonus)

        total_chips += final_chips_p2
        recent_rewards.append(final_reward)
        rolling_results.append(1 if won else 0)

        # ── CSV logging ──────────────────────────────────────────────
        if csv_writer:
            rolling_wr_val = wins / episode
            avg_reward = (sum(recent_rewards) / len(recent_rewards)
                          if recent_rewards else 0)
            current_lr = rl_bot.optimizer.param_groups[0]["lr"]
            csv_writer.writerow([
                episode, int(won), final_reward, rolling_wr_val,
                avg_reward, current_lr,
                "selfplay" if in_selfplay else "mixed",
                f"{heuristic_weight:.2f}", f"{mc_weight:.2f}",
            ])

        # ── Weight shift / self-play transition check ────────────────
        if not in_selfplay and len(rolling_results) >= ROLLING_WINDOW:
            rolling_wr = sum(rolling_results) / len(rolling_results)

            if rolling_wr > SHIFT_THRESHOLD:
                if mc_weight < MAX_MC_WEIGHT:
                    # Shift weights toward MC
                    mc_weight = min(MAX_MC_WEIGHT,
                                    round(mc_weight + SHIFT_STEP, 2))
                    heuristic_weight = round(1.0 - mc_weight, 2)
                    rolling_results.clear()   # reset window after shift
                    print(f"\n{'=' * 70}")
                    print(f"[mixed] WEIGHT SHIFT at episode {episode}  "
                          f"(rolling WR {rolling_wr:.1%})")
                    print(f"  heuristic={heuristic_weight:.0%}  "
                          f"montecarlo={mc_weight:.0%}")
                    print(f"{'=' * 70}\n")
                else:
                    # MC already at max — transition to self-play
                    in_selfplay = True
                    os.makedirs("models", exist_ok=True)
                    rl_bot.save_model("models/rl_selfplay_snapshot.pt")
                    print(f"\n{'=' * 70}")
                    print(f"[mixed] ENTERING SELF-PLAY at episode {episode}  "
                          f"(rolling WR {rolling_wr:.1%})")
                    print(f"  Initial self-play snapshot saved.")
                    print(f"{'=' * 70}\n")

        # ── Progress print ───────────────────────────────────────────
        if episode % 100 == 0:
            rolling_wr = wins / episode
            avg_reward = (sum(recent_rewards) / len(recent_rewards)
                          if recent_rewards else 0)
            current_lr = rl_bot.optimizer.param_groups[0]["lr"]
            mode_str = ("selfplay"
                        if in_selfplay
                        else f"mixed(h={heuristic_weight:.0%}/mc={mc_weight:.0%})")
            print(f"  ep={episode:>6}  wins={wins:>5}  "
                  f"wr={rolling_wr:.1%}  avg_r={avg_reward:+.3f}  "
                  f"lr={current_lr:.1e}  mode={mode_str}")

    # ── End of training ──────────────────────────────────────────────
    rl_bot.flush_buffer()

    os.makedirs("models", exist_ok=True)
    final_path = "models/rl_model_run2.pt"
    rl_bot.save_model(final_path)
    print(f"\nModel saved to {final_path}")

    if csv_file:
        csv_file.close()
        print(f"Training log saved to {csv_path}")

    final_wr = wins / num_episodes if num_episodes > 0 else 0
    avg_final = (sum(recent_rewards) / len(recent_rewards)
                 if recent_rewards else 0)
    print(f"\n{'=' * 70}")
    print(f"Training complete.")
    print(f"  Episodes: {num_episodes}")
    print(f"  Wins:     {wins} / {num_episodes}  ({final_wr:.1%})")
    print(f"  Avg reward (last 100): {avg_final:+.3f}")
    print(f"  Final weights: heuristic={heuristic_weight:.0%}  "
          f"montecarlo={mc_weight:.0%}")
    print(f"  Self-play reached: {'Yes' if in_selfplay else 'No'}")
    print(f"{'=' * 70}")

    return rl_bot


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the RL poker bot with a mixed opponent curriculum"
    )
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Number of tournament episodes (default: 10000)")
    parser.add_argument("--chips", type=int, default=500,
                        help="Starting chips per player (default: 500)")
    parser.add_argument("--opponent", type=str, default="montecarlo",
                        choices=["montecarlo", "heuristic", "self"],
                        help="Fallback opponent type (default: montecarlo)")
    parser.add_argument("--csv", type=str,
                        default="output/rl_training_log_mixed.csv",
                        help="CSV log path (default: output/rl_training_log_mixed.csv)")
    parser.add_argument("--lr_step", type=int, default=30_000,
                        help="Reduce LR by 0.5x every N episodes (default: 30000)")
    parser.add_argument("--snapshot_every", type=int, default=500,
                        help="Self-play snapshot interval (default: 500)")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    train_rl_bot_mixed(
        num_episodes=args.episodes,
        chips_per_player=args.chips,
        opponent_type=args.opponent,
        csv_path=args.csv,
        lr_step_episodes=args.lr_step,
        snapshot_every=args.snapshot_every,
    )
