"""
Train the RL bot through self-play or against other bots.

Supports a curriculum mode that starts against weak opponents and
advances to stronger ones once the rolling win-rate exceeds a threshold.
"""
import os
import sys
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
]

PROMOTE_WIN_RATE = 0.45   # 45 %
PROMOTE_WINDOW   = 500    # rolling window size
MIN_EPISODES     = 10_000 # stay at least this many episodes before first promotion check


def _make_opponent(opponent_type):
    """Build a single-stage opponent from a CLI flag value."""
    if opponent_type == "self":
        return InProcessBot(RLBot(training_mode=False))
    if opponent_type == "montecarlo":
        return PlayerViewAdapter(MonteCarloBot(simulations=200))
    return PlayerViewAdapter(SmartBot())


def train_rl_bot(num_episodes=1000, chips_per_player=500,
                 opponent_type="montecarlo", use_curriculum=False):
    """
    Train RL bot through multiple tournaments.

    Args:
        num_episodes: Number of tournaments to play.
        chips_per_player: Starting chips.
        opponent_type: Fixed opponent when curriculum is off.
        use_curriculum: When True, ignore opponent_type and walk through
                        the three-stage curriculum instead.
    """
    print("=" * 70)
    print("TRAINING RL BOT")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Chip stack per player: {chips_per_player}")
    if use_curriculum:
        stages = " → ".join(s["name"] for s in CURRICULUM)
        print(f"Mode: curriculum ({stages})")
    else:
        print(f"Opponent: {opponent_type}")
    print("=" * 70)
    print()

    # Create RL bot with higher learning rate
    rl_bot = RLBot(training_mode=True, learning_rate=3e-4)

    table = Table()

    wins = 0
    total_chips = 0
    recent_wins = 0

    # Curriculum bookkeeping
    stage_idx = 0
    stage_wins = deque(maxlen=PROMOTE_WINDOW)  # rolling window of 1/0
    stage_episode_count = 0

    if use_curriculum:
        print(f"[curriculum] Starting stage 1/{len(CURRICULUM)}: "
              f"{CURRICULUM[stage_idx]['name']}\n")

    for episode in range(1, num_episodes + 1):
        # Reset for new tournament
        rl_bot.end_episode()
        rl_bot.opponent_stats = {}

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

        while True:
            active_seats = [s for s in seats if s.chips > 0]
            if len(active_seats) <= 1:
                winner = active_seats[0].player_id if active_seats else None
                break

            result = table.play_hand(
                seats=active_seats,
                small_blind=1,
                big_blind=2,
                dealer_index=dealer_index % len(active_seats),
                bot_for={s.player_id: bots[s.player_id] for s in active_seats},
                on_event=None,
            )

            # Record reward normalized to fraction of starting stack
            if "P2" in result:
                reward = result["P2"] / chips_per_player
                rl_bot.record_reward(reward)

            dealer_index = (dealer_index + 1) % len(seats)
            hand_count += 1

            if hand_count > 10000:  # Safety limit
                winner = max(seats, key=lambda s: s.chips).player_id
                break

        # Final reward
        final_chips_p2 = sum(s.chips for s in seats if s.player_id == "P2")
        chip_change = final_chips_p2 - initial_chips_p2
        won = winner == "P2"

        if won:
            wins += 1
            recent_wins += 1

        final_reward = chip_change / chips_per_player
        rl_bot.record_reward(final_reward)
        rl_bot.end_episode()

        total_chips += final_chips_p2

        # ── Curriculum promotion check ───────────────────────────
        if use_curriculum:
            stage_wins.append(1 if won else 0)
            stage_episode_count += 1

            if (stage_idx < len(CURRICULUM) - 1
                    and stage_episode_count >= MIN_EPISODES
                    and len(stage_wins) == PROMOTE_WINDOW):
                rolling_wr = sum(stage_wins) / PROMOTE_WINDOW
                if rolling_wr >= PROMOTE_WIN_RATE:
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
            recent_win_rate = (recent_wins / 50) * 100 if episode >= 50 else win_rate
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
            recent_wins = 0  # Reset recent wins counter

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
                       help="Use curriculum training: random → heuristic → montecarlo")

    args = parser.parse_args()

    train_rl_bot(
        num_episodes=args.episodes,
        chips_per_player=args.chips,
        opponent_type=args.opponent,
        use_curriculum=args.curriculum,
    )
