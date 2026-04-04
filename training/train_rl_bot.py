"""
Train the RL bot through self-play or against other bots.

Supports a curriculum mode that starts against weak opponents and
advances to stronger ones once the rolling win-rate exceeds a threshold.
"""
import os
import sys
import csv
from collections import deque

# Add project root to path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.engine import Table, Seat, InProcessBot, RandomBot
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

# ── Curriculum stages ───────────────────────────────────────────────
CURRICULUM = [
    {"name": "random",      "make_bot": lambda: InProcessBot(RandomBot())},
    {"name": "heuristic",   "make_bot": lambda: PlayerViewAdapter(SmartBot())},
    {"name": "montecarlo",  "make_bot": lambda: PlayerViewAdapter(MonteCarloBot(simulations=200))},
    {"name": "selfplay",    "make_bot": None},  # filled dynamically
]

DEFAULT_PROMOTE_WIN_RATE = 0.55
PROMOTE_WINDOW   = 500    # rolling window size
MIN_EPISODES     = 10_000 # stay at least this many episodes before first promotion check


def _make_opponent(opponent_type):
    """Build a single-stage opponent from a CLI flag value."""
    if opponent_type == "self":
        return InProcessBot(RLBot(training_mode=False))
    if opponent_type == "montecarlo":
        return PlayerViewAdapter(MonteCarloBot(simulations=200))
    return PlayerViewAdapter(SmartBot())


def evaluate_against_mc(rl_bot, num_hands=100, chips=500):
    """Play num_hands against MonteCarloBot and return win rate."""
    table = Table()
    wins = 0
    was_training = rl_bot.training_mode
    rl_bot.training_mode = False
    rl_bot.policy_net.eval()

    for _ in range(num_hands):
        seats = [
            Seat(player_id="P1", chips=chips),
            Seat(player_id="P2", chips=chips),
        ]
        mc_bot = PlayerViewAdapter(MonteCarloBot(simulations=200))
        bots = {"P1": mc_bot, "P2": InProcessBot(rl_bot)}

        hand_count = 0
        dealer_index = 0
        while True:
            active = [s for s in seats if s.chips > 0]
            if len(active) <= 1:
                winner = active[0].player_id if active else None
                break
            table.play_hand(
                seats=active, small_blind=1, big_blind=2,
                dealer_index=dealer_index % len(active),
                bot_for={s.player_id: bots[s.player_id] for s in active},
                on_event=None,
                log_decisions=False,
            )
            dealer_index += 1
            hand_count += 1
            if hand_count > 10000:
                winner = max(seats, key=lambda s: s.chips).player_id
                break
        if winner == "P2":
            wins += 1

    rl_bot.training_mode = was_training
    if was_training:
        rl_bot.policy_net.train()
    return wins / num_hands


def train_rl_bot(num_episodes=1000, chips_per_player=500,
                 opponent_type="montecarlo", use_curriculum=False,
                 promote_win_rate=DEFAULT_PROMOTE_WIN_RATE,
                 csv_path=None, lr_step_episodes=15000):
    """
    Train RL bot through multiple tournaments.

    Args:
        num_episodes: Number of tournaments to play.
        chips_per_player: Starting chips.
        opponent_type: Fixed opponent when curriculum is off.
        use_curriculum: When True, ignore opponent_type and walk through
                        the curriculum stages instead.
        promote_win_rate: Rolling win-rate threshold for curriculum promotion.
        csv_path: If set, write per-episode metrics to this CSV file.
        lr_step_episodes: Reduce LR by 0.5 every this many episodes.
    """
    print("=" * 70)
    print("TRAINING RL BOT")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Chip stack per player: {chips_per_player}")
    print(f"Promotion threshold: {promote_win_rate:.0%}")
    print(f"LR step every: {lr_step_episodes} episodes")
    if use_curriculum:
        stages = " -> ".join(s["name"] for s in CURRICULUM)
        print(f"Mode: curriculum ({stages})")
    else:
        print(f"Opponent: {opponent_type}")
    print("=" * 70)
    print()

    # Create RL bot with higher learning rate
    rl_bot = RLBot(training_mode=True, learning_rate=3e-4)

    # LR scheduler: StepLR with step_size in episodes (manual)
    initial_lr = 3e-4
    lr_decay_factor = 0.5

    table = Table()

    wins = 0
    total_chips = 0
    recent_rewards = deque(maxlen=100)

    # Curriculum bookkeeping
    stage_idx = 0
    stage_wins = deque(maxlen=PROMOTE_WINDOW)  # rolling window of 1/0
    stage_episode_count = 0

    # Baseline for variance reduction (running mean of episode returns)
    baseline_returns = deque(maxlen=1000)

    # Best checkpoint tracking
    best_eval_wr = 0.0

    # CSV logging
    csv_file = None
    csv_writer = None
    if csv_path:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "won", "reward", "rolling_wr", "avg_reward", "lr", "stage"])

    # Self-play bot maker (needs reference to rl_bot)
    CURRICULUM[3]["make_bot"] = lambda: InProcessBot(RLBot(training_mode=False))

    if use_curriculum:
        print(f"[curriculum] Starting stage 1/{len(CURRICULUM)}: "
              f"{CURRICULUM[stage_idx]['name']}\n")

    for episode in range(1, num_episodes + 1):
        # Reset for new tournament
        rl_bot.end_episode()
        rl_bot.opponent_stats = {}

        # ── LR scheduling: reduce every lr_step_episodes ──────────
        if episode > 1 and (episode - 1) % lr_step_episodes == 0:
            num_decays = (episode - 1) // lr_step_episodes
            new_lr = initial_lr * (lr_decay_factor ** num_decays)
            for pg in rl_bot.optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"  [LR] Reduced to {new_lr:.2e} at episode {episode}")

        # ── Build opponent for this episode ──────────────────────
        seats = [
            Seat(player_id="P1", chips=chips_per_player),
            Seat(player_id="P2", chips=chips_per_player),
        ]

        if use_curriculum:
            opponent_bot = CURRICULUM[stage_idx]["make_bot"]()
        else:
            opponent_bot = _make_opponent(opponent_type)

        bots = {
            "P1": opponent_bot,
            "P2": InProcessBot(rl_bot),
        }

        # Play tournament until winner
        hand_count = 0
        dealer_index = 0
        initial_chips_p2 = chips_per_player

        # Per-step rewards for proper credit assignment
        step_rewards = []

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
                log_decisions=False,  # never write logs during RL training
            )

            # Per-hand reward: normalised chip change for this hand only
            chips_after = sum(s.chips for s in seats if s.player_id == "P2")
            hand_reward = (chips_after - chips_before) / chips_per_player
            step_rewards.append(hand_reward)

            # Record this hand's reward to RL bot for credit assignment
            if "P2" in result:
                rl_bot.record_reward(result["P2"] / chips_per_player)

            dealer_index = (dealer_index + 1) % len(seats)
            hand_count += 1

            if hand_count > 10000:  # Safety limit
                winner = max(seats, key=lambda s: s.chips).player_id
                break

        # Final outcome
        final_chips_p2 = sum(s.chips for s in seats if s.player_id == "P2")
        chip_change = final_chips_p2 - initial_chips_p2
        won = winner == "P2"

        if won:
            wins += 1

        final_reward = chip_change / chips_per_player
        # Note: per-hand rewards were already recorded inside the loop above.
        # Do NOT call record_reward again here — that would overwrite every
        # step's reward with the final terminal value, defeating per-step
        # credit assignment.

        total_chips += final_chips_p2
        recent_rewards.append(final_reward)
        baseline_returns.append(final_reward)

        # CSV logging
        if csv_writer:
            rolling_wr_val = wins / episode
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            current_lr = rl_bot.optimizer.param_groups[0]['lr']
            stage_name = CURRICULUM[stage_idx]['name'] if use_curriculum else opponent_type
            csv_writer.writerow([episode, int(won), final_reward, rolling_wr_val,
                                 avg_reward, current_lr, stage_name])

        # ── Curriculum promotion check ───────────────────────────
        if use_curriculum:
            stage_wins.append(1 if won else 0)
            stage_episode_count += 1

            if (stage_idx < len(CURRICULUM) - 1
                    and stage_episode_count >= MIN_EPISODES
                    and len(stage_wins) == PROMOTE_WINDOW):
                rolling_wr = sum(stage_wins) / PROMOTE_WINDOW
                if rolling_wr >= promote_win_rate:
                    stage_idx += 1
                    stage_wins.clear()
                    stage_episode_count = 0
                    print(f"\n{'=' * 70}")
                    print(f"[curriculum] Promoted to stage "
                          f"{stage_idx + 1}/{len(CURRICULUM)}: "
                          f"{CURRICULUM[stage_idx]['name']}  "
                          f"(episode {episode}, rolling WR "
                          f"{rolling_wr:.1%})")
                    print(f"{'=' * 70}\n")

        # Progress update
        if episode % 50 == 0 or episode == num_episodes:
            win_rate = (wins / episode) * 100
            recent_win_rate = (
                sum(1 for r in recent_rewards if r > 0) / len(recent_rewards) * 100
                if recent_rewards else win_rate
            )
            avg_chips = total_chips / episode
            stage_info = ""
            if use_curriculum:
                wr = (sum(stage_wins) / len(stage_wins) * 100
                      if stage_wins else 0)
                stage_info = (f" | Stage: {CURRICULUM[stage_idx]['name']}"
                              f" (WR {wr:.1f}%)")
            print(f"Episode {episode}/{num_episodes} | "
                  f"Win Rate: {win_rate:.1f}% | "
                  f"Recent (50): {recent_win_rate:.1f}% | "
                  f"Avg Final Chips: {avg_chips:.1f}{stage_info}")

    # Process last episode's policy update
    rl_bot.end_episode()

    # Close CSV
    if csv_file:
        csv_file.close()

    # Save trained model
    rl_bot.save_model()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final Win Rate: {(wins/num_episodes)*100:.1f}%")
    if use_curriculum:
        print(f"Final stage: {CURRICULUM[stage_idx]['name']}")
    print(f"Model saved to: models/rl_model.pt")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL bot")
    parser.add_argument("--episodes", type=int, default=50000,
                        help="Number of training episodes (default: 50000)")
    parser.add_argument("--chips", type=int, default=500,
                        help="Starting chips per player")
    parser.add_argument("--opponent", type=str, default="montecarlo",
                        choices=["self", "montecarlo", "smart"],
                        help="Opponent type (ignored when --curriculum is set)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum training: random -> heuristic -> montecarlo")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to write per-episode CSV metrics")

    args = parser.parse_args()

    train_rl_bot(
        num_episodes=args.episodes,
        chips_per_player=args.chips,
        opponent_type=args.opponent,
        use_curriculum=args.curriculum,
        csv_path=args.csv,
    )
