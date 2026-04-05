"""
Train the CFR bot via pure self-play.

A single CFRBot instance is placed on both seats simultaneously.  Because
both players share the same regret table, every hand updates regrets from
both perspectives — the theoretically correct approach for converging toward
a Nash equilibrium strategy.

Convergence note
----------------
In self-play the expected win-rate for each side is ~50 % — that is a sign
of *healthy* convergence, not a bug.  Track ``info_sets`` and
``total_iters`` (printed every 1 000 episodes) to monitor progress.

Checkpoint
----------
* Loads  ``--profile``  (default: models/cfr_regret.pkl) on startup if the
  file already exists.  CFRBot's constructor handles this automatically.
* Saves every ``--save_every`` episodes (default: 500) and at the end of
  training.

Usage
-----
    python training/train_cfr_bot.py
    python training/train_cfr_bot.py --tournaments 100000 --iterations 500
    python training/train_cfr_bot.py --profile models/cfr_v2.pkl --save_every 1000
"""

import os
import sys

# Add project root to path so imports work when run from any directory.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse

from core.engine import Table, Seat, InProcessBot
from core.bot_api import BotAdapter, PlayerView, Action
from bots.cfr_bot import CFRBot


# ---------------------------------------------------------------------------
#  Thin adapter so CFRBot (which doesn't inherit BotAdapter) plugs into the
#  engine's InProcessBot / bot_for dict without modification.
# ---------------------------------------------------------------------------

class _CFRAdapter(BotAdapter):
    """Wraps a CFRBot to satisfy the BotAdapter interface."""

    def __init__(self, bot: CFRBot):
        self.bot = bot

    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)


# ---------------------------------------------------------------------------
#  Main training function
# ---------------------------------------------------------------------------

def train_cfr_bot(
    num_tournaments: int = 50_000,
    chips_per_player: int = 500,
    iterations: int = 200,
    save_every: int = 500,
    profile_path: str = "models/cfr_regret.pkl",
) -> CFRBot:
    """
    Run pure CFR self-play for ``num_tournaments`` episodes.

    Args:
        num_tournaments:  Number of head-up tournament episodes.
        chips_per_player: Starting chip stack for each seat.
        iterations:       MCCFR rollouts per decision point (passed to CFRBot).
        save_every:       Persist the regret table every N episodes.
        profile_path:     Path for regret-table persistence.

    Returns:
        The trained CFRBot instance.
    """
    print("=" * 70)
    print("TRAINING CFR BOT  (pure self-play)")
    print("=" * 70)
    print(f"Episodes:         {num_tournaments}")
    print(f"Chips per player: {chips_per_player}")
    print(f"Iterations/pt:    {iterations}  (MCCFR rollouts per decision)")
    print(f"Save every:       {save_every} episodes")
    print(f"Profile path:     {profile_path}")
    print("=" * 70)
    print()

    # ── Build bot (constructor auto-loads profile if it exists) ──────────────
    bot = CFRBot(
        iterations=iterations,
        profile_path=profile_path,
        use_average=True,
    )

    # Report what was loaded (or that we're starting fresh)
    loaded_stats = bot.stats()
    if loaded_stats["info_sets"] > 0:
        print(
            f"Resumed from {profile_path}: "
            f"{loaded_stats['info_sets']} info sets, "
            f"{loaded_stats['total_iterations']} total iterations.\n"
        )
    else:
        print(f"No existing profile found — starting fresh.\n")

    # ── Shared adapter: same bot instance on both seats ──────────────────────
    # Both P1 and P2 reference the *same* CFRBot object, so every act() call
    # updates the shared regret table from whichever perspective is acting.
    adapter = _CFRAdapter(bot)

    table = Table()
    p1_wins = 0

    # ── Main training loop ───────────────────────────────────────────────────
    for episode in range(1, num_tournaments + 1):

        # Fresh chip stacks each episode
        seats = [
            Seat(player_id="P1", chips=chips_per_player),
            Seat(player_id="P2", chips=chips_per_player),
        ]

        # Both seats use the same CFRBot adapter
        bots = {
            "P1": InProcessBot(adapter),
            "P2": InProcessBot(adapter),
        }

        # ── Play hands until one player is eliminated ────────────────────────
        dealer_index = 0
        hand_count   = 0
        winner       = None

        while True:
            active_seats = [s for s in seats if s.chips > 0]
            if len(active_seats) <= 1:
                winner = active_seats[0].player_id if active_seats else None
                break

            table.play_hand(
                seats=active_seats,
                small_blind=1,
                big_blind=2,
                dealer_index=dealer_index % len(active_seats),
                bot_for={s.player_id: bots[s.player_id] for s in active_seats},
                on_event=None,
                log_decisions=False,
            )

            dealer_index = (dealer_index + 1) % len(seats)
            hand_count  += 1

            if hand_count > 10_000:      # safety cap
                winner = max(seats, key=lambda s: s.chips).player_id
                break

        # ── Episode bookkeeping ──────────────────────────────────────────────
        if winner == "P1":
            p1_wins += 1

        # ── Periodic save ────────────────────────────────────────────────────
        if episode % save_every == 0:
            bot.save(profile_path)

        # ── Progress report every 1 000 episodes ─────────────────────────────
        if episode % 1_000 == 0:
            s = bot.stats()
            p1_wr = p1_wins / episode
            print(
                f"  ep={episode:>7}  "
                f"info_sets={s['info_sets']:<7}  "
                f"total_iters={s['total_iterations']:<10}  "
                f"p1_wr={p1_wr:.1%}"
            )

    # ── End of training ───────────────────────────────────────────────────────
    bot.save(profile_path)

    final_stats = bot.stats()
    final_p1_wr = p1_wins / num_tournaments if num_tournaments > 0 else 0.0

    print(f"\n{'=' * 70}")
    print(f"Training complete.")
    print(f"  Episodes:       {num_tournaments}")
    print(f"  P1 wins:        {p1_wins} / {num_tournaments}  ({final_p1_wr:.1%})")
    print(f"  Info sets:      {final_stats['info_sets']}")
    print(f"  Total iters:    {final_stats['total_iterations']}")
    print(f"  Profile saved:  {profile_path}")
    print(f"{'=' * 70}")

    return bot


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CFR bot via pure self-play (MCCFR)"
    )
    parser.add_argument(
        "--tournaments", type=int, default=50_000,
        help="Number of head-up tournament episodes (default: 50000)"
    )
    parser.add_argument(
        "--chips", type=int, default=500,
        help="Starting chips per player (default: 500)"
    )
    parser.add_argument(
        "--iterations", type=int, default=200,
        help="MCCFR rollouts per decision point (default: 200)"
    )
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="Save regret table every N episodes (default: 500)"
    )
    parser.add_argument(
        "--profile", type=str, default="models/cfr_regret.pkl",
        help="Path for regret-table persistence (default: models/cfr_regret.pkl)"
    )
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    train_cfr_bot(
        num_tournaments=args.tournaments,
        chips_per_player=args.chips,
        iterations=args.iterations,
        save_every=args.save_every,
        profile_path=args.profile,
    )
