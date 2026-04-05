# run_local_match.py — Run a single tournament to completion with CLI control

import argparse
import random
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt

from core.engine import Table, Seat
from bots import parse_players, escalate_blinds


def run_tournament_until_winner(seats, bots, base_sb, base_bb,
                                blind_increase_every, max_hands):
    """Run tournament until one player remains. Returns chip history, hand count,
    elimination order [(player_id, position, hand_eliminated)]."""
    table = Table()
    chip_history = [{s.player_id: s.chips for s in seats}]
    dealer_index = 0
    hand_count = 0
    eliminations: list[tuple[str, int, int]] = []  # (pid, position, hand#)
    total_players = len(seats)

    print("=" * 60)
    print("TOURNAMENT MODE: Playing until one winner!")
    print(f"Blinds: {base_sb}/{base_bb}  |  "
          f"Escalation: every {blind_increase_every} hands (1.5x)"
          if blind_increase_every > 0 else
          f"Blinds: {base_sb}/{base_bb}  |  No escalation")
    print("=" * 60)

    while True:
        active_players = [s for s in seats if s.chips > 0]
        if len(active_players) <= 1:
            if active_players:
                winner = active_players[0].player_id
                print(f"\nTOURNAMENT OVER! Winner: {winner}")
            break

        hand_count += 1
        sb, bb = escalate_blinds(hand_count, base_sb, base_bb, blind_increase_every)

        active_seats = [s for s in seats if s.chips > 0]
        active_bots = {s.player_id: bots[s.player_id] for s in active_seats}

        table.play_hand(
            seats=active_seats,
            small_blind=sb,
            big_blind=bb,
            dealer_index=dealer_index % len(active_seats),
            bot_for=active_bots,
            on_event=None,
        )

        dealer_index = (dealer_index + 1) % len(seats)
        chip_history.append({s.player_id: s.chips for s in seats})

        # Check for new eliminations
        for s in seats:
            if s.chips <= 0 and not any(e[0] == s.player_id for e in eliminations):
                pos = total_players - len(eliminations)
                eliminations.append((s.player_id, pos, hand_count))
                print(f"  [OUT] {s.player_id} eliminated at hand #{hand_count} (position {pos})")

        # Print stacks every 50 hands or if someone was eliminated this hand
        if hand_count % 50 == 0:
            print(f"\n--- Hand #{hand_count} (blinds {sb}/{bb}) ---")
            for s in seats:
                if s.chips > 0:
                    print(f"  {s.player_id}: {s.chips} chips")

        if hand_count >= max_hands:
            print(f"Safety limit reached ({max_hands} hands). Stopping.")
            break

    # Winner gets position 1
    for s in seats:
        if s.chips > 0 and not any(e[0] == s.player_id for e in eliminations):
            eliminations.append((s.player_id, 1, hand_count))

    return chip_history, hand_count, eliminations


def plot_tournament_progress(chip_history, player_ids, bot_types, output_path):
    """Create and save a chip-stack visualization."""
    if not chip_history:
        print("No data to plot.")
        return

    hands = list(range(len(chip_history)))
    plt.figure(figsize=(12, 6))

    for pid in player_ids:
        btype = bot_types.get(pid, "")
        chips_over_time = [state.get(pid, 0) for state in chip_history]
        plt.plot(hands, chips_over_time, label=f"{pid} ({btype})",
                 linewidth=2, markersize=4)

    plt.xlabel("Hand Number", fontsize=12)
    plt.ylabel("Chip Stack", fontsize=12)
    plt.title("Tournament Progress: Chip Stacks Over Time",
              fontsize=14, fontweight="bold")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved as '{output_path}'")
    plt.close()


def print_summary_table(seats, bot_types, eliminations, total_hands):
    """Print a formatted summary table."""
    elim_map = {pid: (pos, hand) for pid, pos, hand in eliminations}

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total hands played: {total_hands}\n")
    print(f"{'Player':<10} {'Bot Type':<15} {'Final Chips':>12} {'Position':>10} {'Hands Survived':>16}")
    print("-" * 70)

    # Sort by finish position
    rows = []
    for s in seats:
        pos, hand_elim = elim_map.get(s.player_id, (0, total_hands))
        rows.append((s.player_id, bot_types.get(s.player_id, "?"),
                      s.chips, pos, hand_elim))
    rows.sort(key=lambda r: r[3])

    for pid, btype, chips, pos, survived in rows:
        pos_str = f"#{pos}" if pos else "?"
        print(f"{pid:<10} {btype:<15} {chips:>12} {pos_str:>10} {survived:>16}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single Texas Hold'em tournament to completion")
    parser.add_argument("--players", type=str,
                        default="mc200,smart,mc100,smart,ml,rl",
                        help="Comma-separated bot types (default: montecarlo,smart,smart,rl)")
    parser.add_argument("--chips", type=int, default=500,
                        help="Starting chips per player")
    parser.add_argument("--sb", type=int, default=1,
                        help="Initial small blind (default: 1)")
    parser.add_argument("--bb", type=int, default=2,
                        help="Initial big blind (default: 2)")
    parser.add_argument("--blind-increase", type=int, default=50,
                        help="Escalate blinds every N hands, 0 to disable (default: 50)")
    parser.add_argument("--max-hands", type=int, default=5000,
                        help="Safety hand limit (default: 5000)")
    parser.add_argument("--output", type=str, default="output/tournament_progress.png",
                        help="Path for the output chart (default: output/tournament_progress.png)")
    parser.add_argument("--rl_model", type=str, default=None,
                        help="Path to RL model weights (e.g. models/rl_model_run3.pt). "
                             "Rewrites any 'rl' entry in --players to use this model.")
    args = parser.parse_args()

    from bots import parse_players, create_bot, escalate_blinds

    if args.rl_model:
        import re
        args.players = re.sub(r'(?<![:\w])rl(?![\w:])', f'rl:{args.rl_model}', args.players)

    player_specs = parse_players(args.players)
    player_ids = [pid for pid, _, _ in player_specs]
    bot_types = {pid: btype for pid, btype, _ in player_specs}

    seats = [Seat(player_id=pid, chips=args.chips) for pid, _, _ in player_specs]
    bots = {pid: create_bot(btype) for pid, btype, _ in player_specs}

    chip_history, total_hands, eliminations = run_tournament_until_winner(
        seats=seats,
        bots=bots,
        base_sb=args.sb,
        base_bb=args.bb,
        blind_increase_every=args.blind_increase,
        max_hands=args.max_hands,
    )

    print_summary_table(seats, bot_types, eliminations, total_hands)
    plot_tournament_progress(chip_history, player_ids, bot_types, args.output)


if __name__ == "__main__":
    main()
