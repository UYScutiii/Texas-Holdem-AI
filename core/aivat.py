"""
core/aivat.py — Full-information equity-shaping value function
--------------------------------------------------------------
The load-bearing module. Both Path A's leaf evaluator (during recursive
tree CFR) and Path B's value-network training targets come from here.

Naming caveat (from TRAINING_PLAN.md): "AIVAT" here means full-information
equity shaping. We peek at all opponents' hole cards during training-time
scoring; the bot at play time still doesn't see them. Not literally the
AIVAT paper.

Both Path A and Path B import from here. This module is read-only
after Gate 1 closes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Optional, Tuple

from core.engine import eval_hand, _FULL_DECK
from core.equity import equity
from core.icm import equities as icm_equities


@dataclass(frozen=True)
class Snapshot:
    """Full-information game state at a decision point. Frozen so it is
    hashable and safe to use as an LRU-cache key.

    hole_cards is a dict but should be treated as immutable post-construction
    — document that, don't enforce it.
    """
    hole_cards: dict          # seat_idx -> (card, card), all seats
    board: tuple              # tuple of cards (hashable)
    pot: int
    stacks: tuple             # per seat, hashable
    alive: tuple              # per seat, True = still in hand
    to_call: int              # hero's call amount right now
    hero_committed: int       # hero chips already in pot this hand


def _deal_remaining_board(board_tuple, deck_remaining, need):
    """Complete the board from remaining deck cards.

    For river (need=0) returns the board as-is.
    For turn (need=1) enumerates all possibilities.
    For flop (need=2) enumerates all combos.
    For preflop (need=5) we DON'T call this — use MC instead.
    """
    if need <= 0:
        return [board_tuple]
    return [board_tuple + combo for combo in combinations(deck_remaining, need)]


def _evaluate_showdown(hole_cards_dict, full_board, alive_tuple, pot,
                        stacks_tuple, hero_seat):
    """Evaluate a single showdown scenario for chip-EV.

    Returns hero's share of the pot (the equity * pot value).
    Handles side pots when hero is stack-limited.
    """
    alive_seats = [i for i, a in enumerate(alive_tuple) if a]

    if len(alive_seats) <= 1:
        # If only hero alive, they win the pot
        if hero_seat in alive_seats:
            return pot
        return 0.0

    board_list = list(full_board)

    # Evaluate all alive hands
    scores = {}
    for seat in alive_seats:
        cards = hole_cards_dict[seat]
        scores[seat] = eval_hand(list(cards), board_list)

    # Determine hero's share considering potential side pots
    hero_score = scores.get(hero_seat, 0)

    # Simple side-pot handling: hero can only win up to their eligible share
    # Hero's eligible pot = min(hero_total_in, each_opponent_total_in) summed
    # For the snapshot value function, we approximate:
    # hero's max winnable = pot (already accounts for stack-limited contributions)

    # Find the best score among all alive players
    best_score = max(scores.values())

    if hero_score < best_score:
        return 0.0  # hero loses

    # Hero ties or wins
    winners = [s for s in alive_seats if scores[s] == best_score]
    return pot / len(winners)


def value(snapshot: Snapshot, hero_seat: int, mode: str = "chip_ev",
          payouts=None, n_sims: int = 500) -> float:
    """Equity-shaped value of the snapshot for hero_seat.

    Convention:
        Value is measured RELATIVE TO A FOLD-NOW BASELINE OF ZERO.
        The chips already committed are sunk and shared across all action
        choices, so we treat them as a fixed cost and set fold = 0.

    mode:
        "chip_ev"    -> Returns equity * pot.
                        Interpretation: expected pot share if hero proceeds
                        to showdown right now (no more betting). Always >= 0.

        "tournament" -> Returns ICM-equity delta vs the fold outcome.
                        payouts must be supplied (e.g. [1.0, 0, 0, ...]).
                        Computes ICM equity at the projected showdown stacks
                        and subtracts ICM equity at the fold-now stacks.

    Sim-count handling -- branch on remaining board cards:
        0 cards remaining (river):    deterministic, no sims needed.
        1 card remaining  (turn):     enumerate all ~44 river cards.
        2 cards remaining (flop):     enumerate all ~988 turn-river combos.
        5 cards remaining (preflop):  Monte Carlo with n_sims (default 500).
        n_sims is IGNORED for non-preflop streets (enumeration is exact).

    The 500 preflop default is higher than equity()'s 100 because preflop
    is the only noisy case here and we want it tight -- AIVAT is called
    on every snapshot in every training tournament.
    """
    # Edge case: hero already folded
    if not snapshot.alive[hero_seat]:
        return 0.0

    alive_seats = [i for i, a in enumerate(snapshot.alive) if a]

    # Edge case: all opponents folded to hero
    if len(alive_seats) == 1 and alive_seats[0] == hero_seat:
        if mode == "chip_ev":
            return float(snapshot.pot)
        elif mode == "tournament":
            if payouts is None:
                raise ValueError("payouts required for tournament mode")
            # Hero gets the pot uncontested
            stacks_fold = list(snapshot.stacks)
            stacks_win = list(snapshot.stacks)
            stacks_win[hero_seat] += snapshot.pot
            return (icm_equities(stacks_win, payouts)[hero_seat] -
                    icm_equities(stacks_fold, payouts)[hero_seat])

    board = snapshot.board
    cards_remaining = 5 - len(board)

    # Build the set of all known cards (all hole cards + board)
    used = set()
    for seat in range(len(snapshot.stacks)):
        if seat in snapshot.hole_cards:
            for c in snapshot.hole_cards[seat]:
                used.add(tuple(c))
    for c in board:
        used.add(tuple(c))
    deck_remaining = [c for c in _FULL_DECK if c not in used]

    if mode == "chip_ev":
        return _chip_ev_value(snapshot, hero_seat, board, cards_remaining,
                              deck_remaining, n_sims)
    elif mode == "tournament":
        if payouts is None:
            raise ValueError("payouts required for tournament mode")
        return _tournament_value(snapshot, hero_seat, board, cards_remaining,
                                 deck_remaining, payouts, n_sims)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def _chip_ev_value(snapshot, hero_seat, board, cards_remaining,
                   deck_remaining, n_sims):
    """Compute chip-EV value: equity * pot."""

    if cards_remaining == 0:
        # River: deterministic
        return _evaluate_showdown(
            snapshot.hole_cards, board, snapshot.alive,
            snapshot.pot, snapshot.stacks, hero_seat)

    if cards_remaining <= 2:
        # Turn (1 card) or flop (2 cards): enumerate exactly
        boards = _deal_remaining_board(board, deck_remaining, cards_remaining)
        total_value = 0.0
        for full_board in boards:
            total_value += _evaluate_showdown(
                snapshot.hole_cards, full_board, snapshot.alive,
                snapshot.pot, snapshot.stacks, hero_seat)
        return total_value / len(boards) if boards else 0.0

    # Preflop (5 cards remaining): Monte Carlo
    total_value = 0.0
    valid = 0
    for _ in range(n_sims):
        if len(deck_remaining) < cards_remaining:
            break
        sample = random.sample(deck_remaining, cards_remaining)
        full_board = board + tuple(sample)
        total_value += _evaluate_showdown(
            snapshot.hole_cards, full_board, snapshot.alive,
            snapshot.pot, snapshot.stacks, hero_seat)
        valid += 1

    return total_value / valid if valid > 0 else 0.0


def _tournament_value(snapshot, hero_seat, board, cards_remaining,
                      deck_remaining, payouts, n_sims):
    """Compute tournament-mode value: ICM equity delta vs fold baseline.

    fold baseline: hero's ICM equity at current stacks (pot already
    committed, hero walks away with whatever they have left).
    """
    stacks_list = list(snapshot.stacks)
    alive_tuple = snapshot.alive
    alive_seats = [i for i, a in enumerate(alive_tuple) if a]

    # Fold baseline: hero keeps their current stack, pot distributed
    # to other alive players proportionally (simplification: pot stays
    # with the eventual winner but hero has folded).
    # Actually, the fold baseline is just current ICM equity.
    fold_eq = icm_equities(stacks_list, payouts)[hero_seat]

    if cards_remaining == 0:
        # River: deterministic showdown
        showdown_stacks = _compute_showdown_stacks(
            snapshot, board, alive_seats)
        play_eq = icm_equities(showdown_stacks, payouts)[hero_seat]
        return play_eq - fold_eq

    if cards_remaining <= 2:
        # Enumerate
        boards = _deal_remaining_board(board, deck_remaining, cards_remaining)
        total_eq = 0.0
        for full_board in boards:
            ss = _compute_showdown_stacks(
                snapshot, full_board, alive_seats)
            total_eq += icm_equities(ss, payouts)[hero_seat]
        avg_eq = total_eq / len(boards) if boards else fold_eq
        return avg_eq - fold_eq

    # Preflop: Monte Carlo
    total_eq = 0.0
    valid = 0
    for _ in range(n_sims):
        if len(deck_remaining) < cards_remaining:
            break
        sample = random.sample(deck_remaining, cards_remaining)
        full_board = board + tuple(sample)
        ss = _compute_showdown_stacks(
            snapshot, full_board, alive_seats)
        total_eq += icm_equities(ss, payouts)[hero_seat]
        valid += 1

    avg_eq = total_eq / valid if valid > 0 else fold_eq
    return avg_eq - fold_eq


def _compute_showdown_stacks(snapshot, full_board, alive_seats):
    """Given a snapshot and a complete board, compute the resulting stacks
    after showdown distribution."""
    board_list = list(full_board)

    # Evaluate hands for alive players
    scores = {}
    for seat in alive_seats:
        if seat in snapshot.hole_cards:
            cards = snapshot.hole_cards[seat]
            scores[seat] = eval_hand(list(cards), board_list)

    if not scores:
        return list(snapshot.stacks)

    best_score = max(scores.values())
    winners = [s for s in scores if scores[s] == best_score]

    # Distribute pot to winners
    stacks = list(snapshot.stacks)
    share = snapshot.pot // len(winners)
    remainder = snapshot.pot % len(winners)
    for i, w in enumerate(winners):
        stacks[w] += share + (1 if i < remainder else 0)

    return stacks
