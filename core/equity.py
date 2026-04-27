"""
core/equity.py — Canonical multiway equity calculator
-----------------------------------------------------
Single source of truth for Monte Carlo equity estimation.
Extracted from bots/cfr_bot.py (session 2026-04-26).

Both Path A and Path B import from here. This module is read-only
after Gate 1 closes.
"""
from __future__ import annotations

import random
from itertools import combinations
from typing import List, Tuple

from core.engine import eval_hand, _FULL_DECK


def equity(hole_cards, board, n_opponents, n_sims=100):
    """Hero's expected fraction of pot vs n_opponents random hands.

    Hero wins -> 1.0. Tie at showdown -> 0.5 / (number of tying players).
    Hero loses -> 0.0. Returned value is mean over n_sims trials.

    Parameters
    ----------
    hole_cards : list of (rank, suit) tuples
        Hero's two hole cards.
    board : list of (rank, suit) tuples
        Community cards dealt so far (0-5 cards).
    n_opponents : int
        Number of opponents (required, no default). Must be >= 1.
    n_sims : int
        Number of Monte Carlo simulations to run.
    """
    if not hole_cards or len(hole_cards) < 2:
        return 0.5

    n_opponents = max(1, n_opponents)

    used = set(tuple(c) for c in hole_cards) | set(tuple(c) for c in board)
    remaining = [c for c in _FULL_DECK if c not in used]

    wins = 0.0
    total = 0

    for _ in range(n_sims):
        # Need 2 cards per opponent
        if len(remaining) < 2 * n_opponents:
            break

        # Deal cards to all opponents from the remaining deck
        sampled = random.sample(remaining, 2 * n_opponents)
        opp_hands = [sampled[i * 2:(i + 1) * 2] for i in range(n_opponents)]

        # Remove dealt opponent cards before completing the board
        dealt_set = set(tuple(c) for c in sampled)
        rest = [c for c in remaining if tuple(c) not in dealt_set]

        # Complete board to 5 cards if needed
        need = 5 - len(board)
        if need > 0:
            if len(rest) < need:
                continue
            extra = random.sample(rest, need)
            full_board = list(board) + extra
        else:
            full_board = list(board)

        my_score = eval_hand(list(hole_cards), full_board)

        # Hero must beat ALL opponents to win; tie for best = split
        best_opp = max(eval_hand(oh, full_board) for oh in opp_hands)
        if my_score > best_opp:
            wins += 1
        elif my_score == best_opp:
            # Count how many opponents tied with hero
            tying = sum(1 for oh in opp_hands
                        if eval_hand(oh, full_board) == my_score)
            wins += 1.0 / (1 + tying)  # hero's share of the tie
        total += 1

    return wins / total if total > 0 else 0.5


def equity_bucket(hole_cards, board, n_opponents, n_buckets=20, n_sims=100):
    """Integer bucket [0, n_buckets) for hero's equity.

    Used for Path A info-set keying.

    Parameters
    ----------
    hole_cards : list of (rank, suit) tuples
    board : list of (rank, suit) tuples
    n_opponents : int (required, no default)
    n_buckets : int
    n_sims : int
    """
    if not hole_cards or len(hole_cards) < 2 or not board:
        return n_buckets // 2  # neutral bucket when no board

    eq = equity(hole_cards, board, n_opponents, n_sims=n_sims)
    bucket = int(eq * n_buckets)
    return max(0, min(n_buckets - 1, bucket))
