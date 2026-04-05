"""
Train the RL bot through a three-stage curriculum:
    random → heuristic → self-play

Curriculum rules
----------------
* Promotion thresholds (rolling window = 1 000 episodes):
    - random   → heuristic : 60 % WR
    - heuristic → self-play : 55 % WR
* No demotion: once promoted, the bot stays at the new stage.
* Self-play snapshots are saved every 500 episodes to
    models/rl_selfplay_snapshot.pt
  and the opponent always loads from that frozen snapshot.

Checkpoint
----------
* Loads  models/rl_model_run2.pt  if it exists, otherwise starts fresh.
* Saves  models/rl_model_run3.pt  at the end of training.

Log
---
* CSV written to  output/rl_training_log_selfplay.csv
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
from bots.poker_mind_bot import SmartBot
import argparse
import torch


class PlayerViewAdapter(BotAdapter):
    """Adapter for bots that expect a PlayerView."""
    def __init__(self, bot):
        self.bot = bot

    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)


# ── Curriculum definition ────────────────────────────────────────────────────
# Three stages: random → heuristic → self-play.
# The self-play entry has make_bot=None; it is patched inside train_rl_bot().
CURRICULUM = [
    {
        "name":       "random",
        "make_bot":   lambda: InProcessBot(RandomBot()),
        "promote_wr": 0.60,
    },
    {
        "name":       "heuristic",
        "make_bot":   lambda: PlayerViewAdapter(SmartBot()),
        "promote_wr": 0.55,
    },
    {
        "name":       "selfplay",
        "make_bot":   None,          # filled dynamically below
    },
]

PROMOTE_WINDOW   = 1_000   # rolling window for promotion check
SNAPSHOT_PATH    = "models/rl_selfplay_snapshot.pt"
SNAPSHOT_EVERY   = 500     # save a new snapshot every N episodes in self-play
START_CHECKPOINT = "models/rl_model_run2.pt"
FINAL_MODEL_PATH = "models/rl_model_run3.pt"
DEFAULT_CSV_PATH = "output/rl_training_log_selfplay.csv"
HIDDEN_SIZE      = 512


# ── Main training function ───────────────────────────────────────────────────

def train_rl_bot(num_episodes=10_000, chips_per_player=500,
                 csv_path=None, lr_step_episodes=30_000):
    """
    Train RL bot through the three-stage curriculum (no demotion).

    Args:
        num_episodes:    Number of tournament episodes.
        chips_per_player: Starting chips per player.
        csv_path:        Optional path to write per-episode CSV log.
        lr_step_episodes: Reduce LR by 0.5× every this many episodes.
    """
    print("=" * 70)
    print("TRAINING RL BOT  (selfplay curriculum)")
    print("=" * 70)
    print(f"Episodes:            {num_episodes}")
    print(f"Chips per player:    {chips_per_player}")
    print(f"Hidden size:         {HIDDEN_SIZE}")
    print(f"Curriculum:          {' -> '.join(s['name'] for s in CURRICULUM)}")
    thresholds = ", ".join(
        f"{s['name']}={s['promote_wr']:.0%}"
        for s in CURRICULUM if "promote_wr" in s
    )
    print(f"Promotion thresholds:{thresholds}  (over {PROMOTE_WINDOW} episodes)")
    print(f"Demotion:            disabled")
    print(f"Start checkpoint:    {START_CHECKPOINT}")
    print(f"Final model:         {FINAL_MODEL_PATH}")
    print(f"LR step every:       {lr_step_episodes} episodes")
    print(f"Snapshot every:      {SNAPSHOT_EVERY} episodes (self-play stage)")
    print("=" * 70)
    print()

    # ── Build RL bot (512-unit networks via rl_bot.py) ───────────────────
    # RLBot now constructs at hidden=512 directly; no post-construction swap.
    # We construct without model_path so the default load attempt doesn't
    # crash on a 256-unit checkpoint; we do our own graceful load below.
    rl_bot = RLBot(
        model_path="",           # skip the internal auto-load
        training_mode=True,
        learning_rate=3e-4,
    )
    # Graceful checkpoint load: 256-unit weights won't match 512-unit networks.
    # On any error (missing file OR size mismatch) just start fresh.
    if os.path.exists(START_CHECKPOINT):
        try:
            checkpoint = torch.load(START_CHECKPOINT, map_location=rl_bot.device)
            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                rl_bot.policy_net.load_state_dict(checkpoint['policy'])
                rl_bot.value_net.load_state_dict(checkpoint['value'])
            else:
                rl_bot.policy_net.load_state_dict(checkpoint)
            rl_bot.policy_net.train()
            rl_bot.value_net.train()
            print(f"Loaded checkpoint from {START_CHECKPOINT}")
        except RuntimeError as e:
            print(f"[checkpoint] Size mismatch loading {START_CHECKPOINT} "
                  f"(probably 256-unit weights vs 512-unit network) — starting fresh.\n"
                  f"  Detail: {e}")
        except Exception as e:
            print(f"[checkpoint] Could not load {START_CHECKPOINT}: {e} — starting fresh")
    else:
        print(f"[checkpoint] {START_CHECKPOINT} not found — starting fresh")

    initial_lr      = 3e-4
    lr_decay_factor = 0.5

    # ── Pre-build self-play opponent placeholder ──────────────────────────
    # The actual opponent is constructed once per episode (or once per
    # snapshot interval) inside the training loop — NOT via a lambda that
    # fires on every hand.  We keep a reference here and rebuild it only
    # when a fresh snapshot has been saved.
    selfplay_opponent: InProcessBot | None = None

    table = Table()

    wins           = 0
    recent_rewards = deque(maxlen=100)

    # Curriculum bookkeeping
    stage_idx           = 0
    stage_wins          = deque(maxlen=PROMOTE_WINDOW)
    stage_episode_count = 0

    # CSV logging setup
    csv_file   = None
    csv_writer = None
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        csv_file   = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["episode", "won", "reward", "rolling_wr", "avg_reward", "lr", "stage"]
        )

    print(f"[curriculum] Starting stage 1/{len(CURRICULUM)}: "
          f"{CURRICULUM[stage_idx]['name']}\n")

    # ── Main training loop ────────────────────────────────────────────────
    for episode in range(1, num_episodes + 1):

        # Reset episode state
        rl_bot.end_episode()
        rl_bot.opponent_stats = {}

        # ── LR scheduling ────────────────────────────────────────────────
        if episode > 1 and (episode - 1) % lr_step_episodes == 0:
            num_decays = (episode - 1) // lr_step_episodes
            new_lr = initial_lr * (lr_decay_factor ** num_decays)
            for pg in rl_bot.optimizer.param_groups:
                pg["lr"] = new_lr
            print(f"  [LR] Reduced to {new_lr:.2e} at episode {episode}")

        # ── Self-play: save snapshot, then rebuild opponent if needed ─────
        # Snapshot saved every SNAPSHOT_EVERY episodes; opponent rebuilt
        # only when the snapshot changes (or first time we enter self-play).
        in_selfplay = stage_idx == len(CURRICULUM) - 1
        if in_selfplay:
            snapshot_due = (episode % SNAPSHOT_EVERY == 0)
            if snapshot_due:
                os.makedirs("models", exist_ok=True)
                rl_bot.save_model(SNAPSHOT_PATH)
                # Force a rebuild from the new snapshot on this episode
                selfplay_opponent = None

            if selfplay_opponent is None and os.path.exists(SNAPSHOT_PATH):
                selfplay_opponent = InProcessBot(
                    RLBot(
                        model_path=SNAPSHOT_PATH,
                        training_mode=False,
                        use_fallback=True,
                    )
                )

        # ── Build opponent once per episode (reused for all hands) ────────
        seats = [
            Seat(player_id="P1", chips=chips_per_player),
            Seat(player_id="P2", chips=chips_per_player),
        ]
        if in_selfplay and selfplay_opponent is not None:
            opponent_bot = selfplay_opponent
        else:
            opponent_bot = CURRICULUM[stage_idx]["make_bot"]()
        bots = {
            "P1": opponent_bot,
            "P2": InProcessBot(rl_bot),
        }

        # ── Play tournament until one player is eliminated ────────────────
        hand_count      = 0
        dealer_index    = 0
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

            # Per-hand reward: normalised chip delta for proper credit assignment
            chips_after = sum(s.chips for s in seats if s.player_id == "P2")
            if "P2" in result:
                rl_bot.record_reward(
                    (chips_after - chips_before) / max(chips_before, 1)
                )

            dealer_index = (dealer_index + 1) % len(seats)
            hand_count  += 1

            if hand_count > 10_000:   # safety limit
                winner = max(seats, key=lambda s: s.chips).player_id
                break

        # ── Episode outcome ───────────────────────────────────────────────
        final_chips_p2 = sum(s.chips for s in seats if s.player_id == "P2")
        won            = winner == "P2"
        final_reward   = (final_chips_p2 - initial_chips_p2) / max(initial_chips_p2, 1)

        # Terminal bonus: global win/loss signal on top of per-hand rewards
        final_bonus = 1.0 if won else -0.5
        rl_bot.record_reward(final_bonus)

        if won:
            wins += 1
        recent_rewards.append(final_reward)

        # ── CSV row ───────────────────────────────────────────────────────
        if csv_writer:
            rolling_wr_val = wins / episode
            avg_reward     = sum(recent_rewards) / len(recent_rewards)
            current_lr     = rl_bot.optimizer.param_groups[0]["lr"]
            csv_writer.writerow([
                episode, int(won), final_reward,
                rolling_wr_val, avg_reward,
                current_lr, CURRICULUM[stage_idx]["name"]
            ])

        # ── Curriculum promotion check (no demotion) ──────────────────────
        stage_wins.append(1 if won else 0)
        stage_episode_count += 1

        if stage_idx < len(CURRICULUM) - 1 and len(stage_wins) >= PROMOTE_WINDOW:
            rolling_wr     = sum(stage_wins) / len(stage_wins)
            promote_thresh = CURRICULUM[stage_idx]["promote_wr"]
            if rolling_wr >= promote_thresh:
                stage_idx          += 1
                stage_episode_count = 0
                stage_wins.clear()
                print(f"\n{'=' * 70}")
                print(f"[curriculum] PROMOTED to stage "
                      f"{stage_idx + 1}/{len(CURRICULUM)}: "
                      f"{CURRICULUM[stage_idx]['name']}  "
                      f"(episode {episode}, rolling WR {rolling_wr:.1%})")
                print(f"{'=' * 70}\n")
                # Write initial snapshot when entering self-play,
                # and force the opponent to be rebuilt next episode.
                if stage_idx == len(CURRICULUM) - 1:
                    os.makedirs("models", exist_ok=True)
                    rl_bot.save_model(SNAPSHOT_PATH)
                    selfplay_opponent = None   # will be built at top of next episode
                    print(f"  [snapshot] Initial self-play snapshot saved.\n")
            elif episode % 1_000 == 0:
                # Periodic diagnostic: why haven't we promoted?
                needed = CURRICULUM[stage_idx]["promote_wr"]
                print(f"  [curriculum] stage={CURRICULUM[stage_idx]['name']}  "
                      f"rolling_wr={rolling_wr:.1%}/{needed:.0%}  "
                      f"stage_eps={stage_episode_count}")

        # ── Progress print every 100 episodes ────────────────────────────
        if episode % 100 == 0:
            rolling_wr_display = wins / episode
            avg_reward         = sum(recent_rewards) / len(recent_rewards)
            current_lr         = rl_bot.optimizer.param_groups[0]["lr"]
            stage_name         = CURRICULUM[stage_idx]["name"]
            print(f"  ep={episode:>6}  wins={wins:>5}  "
                  f"wr={rolling_wr_display:.1%}  avg_r={avg_reward:+.3f}  "
                  f"lr={current_lr:.1e}  stage={stage_name}")

    # ── End of training ───────────────────────────────────────────────────
    rl_bot.flush_buffer()

    os.makedirs("models", exist_ok=True)
    rl_bot.save_model(FINAL_MODEL_PATH)
    print(f"\nModel saved to {FINAL_MODEL_PATH}")

    if csv_file:
        csv_file.close()
        print(f"Training log saved to {csv_path}")

    final_wr  = wins / num_episodes if num_episodes > 0 else 0
    avg_final = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
    print(f"\n{'=' * 70}")
    print(f"Training complete.")
    print(f"  Episodes:            {num_episodes}")
    print(f"  Wins:                {wins} / {num_episodes}  ({final_wr:.1%})")
    print(f"  Avg reward (last 100): {avg_final:+.3f}")
    print(f"{'=' * 70}")

    return rl_bot


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL poker bot: random → heuristic → self-play curriculum"
    )
    parser.add_argument(
        "--episodes", type=int, default=50_000,
        help="Number of tournament episodes to train for (default: 50000)"
    )
    parser.add_argument(
        "--chips", type=int, default=500,
        help="Starting chips per player (default: 500)"
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV_PATH,
        help=f"Path to write per-episode CSV log (default: {DEFAULT_CSV_PATH})"
    )
    parser.add_argument(
        "--lr_step", type=int, default=30_000,
        help="Reduce LR by 0.5× every this many episodes (default: 30000)"
    )
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    train_rl_bot(
        num_episodes=args.episodes,
        chips_per_player=args.chips,
        csv_path=args.csv,
        lr_step_episodes=args.lr_step,
    )
