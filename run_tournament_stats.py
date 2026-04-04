# run_tournament_stats.py — Run multiple tournaments and track detailed statistics

import argparse
import csv
import io
import os
import random
from collections import defaultdict
from contextlib import redirect_stdout
from multiprocessing import Pool

from core.engine import Table, Seat
from bots import parse_players, escalate_blinds, create_bot


# ── Single silent tournament ──────────────────────────────────────────────────

def run_silent_tournament(args_tuple):
    """Run one tournament silently. Accepts a tuple for multiprocessing.Pool.map.

    Returns dict with: winner, hand_count, finish_order [(pid, position, hand#, chips_at_elim)].
    """
    player_specs, chips, base_sb, base_bb, blind_increase_every, max_hands, seed = args_tuple

    if seed is not None:
        random.seed(seed)

    # Rebuild bots in this process (can't pickle adapters across processes)
    bots = {}
    for pid, btype, _ in player_specs:
        bots[pid] = create_bot(btype)

    seats = [Seat(player_id=pid, chips=chips) for pid, _, _ in player_specs]
    table = Table()
    dealer_index = 0
    hand_count = 0
    total_players = len(seats)
    finish_order: list[tuple[str, int, int, int]] = []  # (pid, pos, hand#, chips_at_elim)

    with redirect_stdout(io.StringIO()):
        while True:
            active_players = [s for s in seats if s.chips > 0]
            if len(active_players) <= 1:
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

            # Track eliminations
            for s in seats:
                if s.chips <= 0 and not any(e[0] == s.player_id for e in finish_order):
                    pos = total_players - len(finish_order)
                    finish_order.append((s.player_id, pos, hand_count, 0))

            if hand_count >= max_hands:
                break

    # Winner / survivors
    for s in seats:
        if s.chips > 0 and not any(e[0] == s.player_id for e in finish_order):
            finish_order.append((s.player_id, 1, hand_count, s.chips))

    # If we hit max_hands, assign remaining positions by chip count
    unfinished = [s for s in seats
                  if not any(e[0] == s.player_id for e in finish_order)]
    unfinished.sort(key=lambda s: s.chips, reverse=True)
    next_pos = total_players - len(finish_order)
    for s in unfinished:
        finish_order.append((s.player_id, next_pos, hand_count, s.chips))
        next_pos -= 1

    winner = None
    for pid, pos, _, _ in finish_order:
        if pos == 1:
            winner = pid
            break
    if winner is None and finish_order:
        winner = finish_order[-1][0]

    return {
        "winner": winner,
        "hand_count": hand_count,
        "finish_order": finish_order,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_tournament_batch(player_spec_str, num_tournaments, chips, base_sb, base_bb,
                         blind_increase_every, max_hands, parallel, output_csv, seed):
    player_specs = parse_players(player_spec_str)
    if len(player_specs) < 2:
        print("Error: need at least 2 players.")
        return

    pids = [pid for pid, _, _ in player_specs]
    bot_types = {pid: btype for pid, btype, _ in player_specs}

    print("=" * 75)
    print(f"RUNNING {num_tournaments} TOURNAMENTS")
    print("=" * 75)
    print(f"Players: {', '.join(f'{pid}={btype}' for pid, btype, _ in player_specs)}")
    print(f"Chips: {chips}  |  Blinds: {base_sb}/{base_bb}  |  "
          f"Escalation every {blind_increase_every} hands")
    if parallel > 1:
        print(f"Parallel workers: {parallel}")
    print("=" * 75)
    print()

    # Build args tuples for each tournament
    tasks = []
    for i in range(num_tournaments):
        t_seed = (seed + i) if seed is not None else None
        tasks.append((player_specs, chips, base_sb, base_bb,
                      blind_increase_every, max_hands, t_seed))

    # Run tournaments
    results = []
    if parallel > 1:
        with Pool(processes=parallel) as pool:
            for i, res in enumerate(pool.imap_unordered(run_silent_tournament, tasks), 1):
                results.append(res)
                if i % 5 == 0 or i == num_tournaments:
                    print(f"  Completed {i}/{num_tournaments}...")
    else:
        for i, task in enumerate(tasks, 1):
            res = run_silent_tournament(task)
            results.append(res)
            if i % 5 == 0 or i == num_tournaments:
                winner = res["winner"]
                hands = res["hand_count"]
                print(f"  Tournament {i}/{num_tournaments} — Winner: {winner} ({hands} hands)")

    # ── Aggregate statistics ──────────────────────────────────────────────────

    wins = defaultdict(int)
    finish_positions = defaultdict(list)       # pid -> [positions]
    chips_at_elimination = defaultdict(list)   # pid -> [chips when eliminated]
    hands_survived = defaultdict(list)         # pid -> [hand# when eliminated]
    h2h_wins = defaultdict(lambda: defaultdict(int))  # pid_a -> pid_b -> count a beat b

    hand_counts = []

    for res in results:
        hand_counts.append(res["hand_count"])
        wins[res["winner"]] += 1

        fo = res["finish_order"]  # [(pid, pos, hand#, chips)]

        for pid, pos, hand, elim_chips in fo:
            finish_positions[pid].append(pos)
            hands_survived[pid].append(hand)
            if pos > 1:
                chips_at_elimination[pid].append(elim_chips)

            # Head-to-head: count wins against each opponent who finished worse
            for other_pid, other_pos, _, _ in fo:
                if other_pid != pid and pos < other_pos:
                    h2h_wins[pid][other_pid] += 1

    # ── Print results ─────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("LEADERBOARD (sorted by win rate)")
    print("=" * 75)

    header = (f"{'#':<4} {'Player':<8} {'Bot':<14} {'Wins':>6} {'Win%':>7} "
              f"{'Avg Pos':>8} {'Avg Elim $':>10} "
              f"{'Hands (avg)':>12} {'(min)':>7} {'(max)':>7}")
    print(header)
    print("-" * 75)

    # Sort by win rate desc
    sorted_pids = sorted(pids, key=lambda p: wins[p], reverse=True)

    for rank, pid in enumerate(sorted_pids, 1):
        btype = bot_types[pid]
        w = wins[pid]
        wr = (w / num_tournaments) * 100
        avg_pos = sum(finish_positions[pid]) / len(finish_positions[pid])
        elim_chips = chips_at_elimination[pid]
        avg_elim = sum(elim_chips) / len(elim_chips) if elim_chips else 0
        hs = hands_survived[pid]
        avg_h = sum(hs) / len(hs) if hs else 0
        min_h = min(hs) if hs else 0
        max_h = max(hs) if hs else 0

        print(f"{rank:<4} {pid:<8} {btype:<14} {w:>6} {wr:>6.1f}% "
              f"{avg_pos:>8.2f} {avg_elim:>10.0f} "
              f"{avg_h:>12.1f} {min_h:>7} {max_h:>7}")

    # Hand count stats
    print(f"\n{'Tournaments:':<25} {num_tournaments}")
    print(f"{'Avg hands/tournament:':<25} {sum(hand_counts)/len(hand_counts):.1f}")
    print(f"{'Shortest:':<25} {min(hand_counts)}")
    print(f"{'Longest:':<25} {max(hand_counts)}")

    # Head-to-head matrix
    print("\n" + "=" * 75)
    print("HEAD-TO-HEAD WIN RATES")
    print("=" * 75)

    # Header row
    col_w = 10
    print(f"{'':>{col_w}}", end="")
    for pid in pids:
        print(f"{pid:>{col_w}}", end="")
    print()

    for pid_a in pids:
        print(f"{pid_a:>{col_w}}", end="")
        for pid_b in pids:
            if pid_a == pid_b:
                print(f"{'---':>{col_w}}", end="")
            else:
                total = h2h_wins[pid_a][pid_b] + h2h_wins[pid_b][pid_a]
                if total > 0:
                    rate = (h2h_wins[pid_a][pid_b] / total) * 100
                    print(f"{rate:>{col_w - 1}.0f}%", end="")
                else:
                    print(f"{'N/A':>{col_w}}", end="")
        print()

    print("=" * 75)

    # ── CSV output ────────────────────────────────────────────────────────────

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tournament", "winner", "hands",
                             *[f"{pid}_position" for pid in pids],
                             *[f"{pid}_hands_survived" for pid in pids]])
            for i, res in enumerate(results, 1):
                fo = {pid: (pos, hand) for pid, pos, hand, _ in res["finish_order"]}
                row = [i, res["winner"], res["hand_count"]]
                for pid in pids:
                    pos, hand = fo.get(pid, (0, 0))
                    row.append(pos)
                for pid in pids:
                    pos, hand = fo.get(pid, (0, 0))
                    row.append(hand)
                writer.writerow(row)
        print(f"\nResults saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple Texas Hold'em tournaments and track statistics")
    parser.add_argument("--players", type=str,
                        default="rl,ml,smart,mc200,mc100",
                        help="Comma-separated bot types (default: rl,ml,smart,mc200,mc100)")
    parser.add_argument("--tournaments", type=int, default=30,
                        help="Number of tournaments (default: 30)")
    parser.add_argument("--chips", type=int, default=500,
                        help="Starting chips per player (default: 500)")
    parser.add_argument("--sb", type=int, default=1,
                        help="Starting small blind (default: 1)")
    parser.add_argument("--bb", type=int, default=2,
                        help="Starting big blind (default: 2)")
    parser.add_argument("--blind-increase-every", type=int, default=50,
                        help="Increase blinds 1.5x every N hands, 0 to disable (default: 50)")
    parser.add_argument("--max-hands", type=int, default=10000,
                        help="Safety hand limit per tournament (default: 10000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1, sequential)")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Save per-tournament results to CSV file")
    args = parser.parse_args()

    run_tournament_batch(
        player_spec_str=args.players,
        num_tournaments=args.tournaments,
        chips=args.chips,
        base_sb=args.sb,
        base_bb=args.bb,
        blind_increase_every=args.blind_increase_every,
        max_hands=args.max_hands,
        parallel=args.parallel,
        output_csv=args.output_csv,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
