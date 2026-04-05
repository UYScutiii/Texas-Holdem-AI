"""
bots/opponent_model_bot.py — Bayesian Opponent-Modeling Bot

Maintains a lightweight Bayesian hand-range estimate for each opponent
using five strength buckets (trash / weak / medium / strong / premium).
Opponent actions update the distribution via likelihood multipliers.
Own decisions are based on Monte-Carlo equity versus the estimated
opponent range rather than random hands.
"""

import random
from typing import Optional
import numpy as np
from core.bot_api import Action, PlayerView
from core.engine import eval_hand, EVAL_HAND_MAX, RANKS, SUITS

# ── Hand-strength buckets ────────────────────────────────────────────────────
# Normalised eval_hand score thresholds (0.0–1.0)
BUCKET_NAMES = ("trash", "weak", "medium", "strong", "premium")
BUCKET_THRESHOLDS = (0.20, 0.40, 0.60, 0.80)  # upper bounds for each bucket (exclusive)
NUM_BUCKETS = len(BUCKET_NAMES)

# ── Likelihood tables ────────────────────────────────────────────────────────
# Each table maps action → per-bucket multiplier [trash, weak, medium, strong, premium]
# Values > 1 mean the action makes that bucket *more* likely.
# Street-specific tables allow context-sensitive updates.

PREFLOP_LIKELIHOODS = {
    "fold":  np.array([2.0, 1.5, 0.8, 0.3, 0.1]),
    "check": np.array([1.5, 1.3, 1.0, 0.7, 0.5]),
    "call":  np.array([0.5, 1.0, 1.5, 1.2, 0.8]),
    "bet":   np.array([0.3, 0.5, 1.0, 1.5, 2.0]),
    "raise": np.array([0.1, 0.3, 0.7, 1.5, 2.5]),
}

POSTFLOP_LIKELIHOODS = {
    "fold":  np.array([2.5, 1.5, 0.6, 0.2, 0.05]),
    "check": np.array([1.2, 1.4, 1.2, 0.8, 0.4]),
    "call":  np.array([0.4, 1.0, 1.5, 1.3, 0.8]),
    "bet":   np.array([0.2, 0.5, 1.0, 1.6, 2.0]),
    "raise": np.array([0.1, 0.2, 0.6, 1.5, 3.0]),
}


def _uniform_prior() -> np.ndarray:
    """Return a uniform distribution over the 5 buckets."""
    return np.ones(NUM_BUCKETS, dtype=np.float64) / NUM_BUCKETS


def _normalise(dist: np.ndarray) -> np.ndarray:
    """Normalise a distribution so it sums to 1. Falls back to uniform."""
    total = dist.sum()
    if total <= 0:
        return _uniform_prior()
    return dist / total


def _hand_bucket(hole, board) -> int:
    """Map a concrete hand + board to a bucket index (0–4)."""
    score = eval_hand(hole, board)
    strength = score / EVAL_HAND_MAX
    for i, thresh in enumerate(BUCKET_THRESHOLDS):
        if strength < thresh:
            return i
    return NUM_BUCKETS - 1


def _preflop_bucket_from_hole(hole) -> int:
    """Classify a 2-card hand into a bucket index using a simple heuristic."""
    rank_val = {r: i for i, r in enumerate(RANKS)}
    r1, r2 = rank_val[hole[0][0]], rank_val[hole[1][0]]
    high, low = max(r1, r2), min(r1, r2)
    suited = hole[0][1] == hole[1][1]
    pair = r1 == r2

    # Premium (bucket 4): AA, KK, QQ, AKs
    if pair and high >= 10:  # QQ+
        return 4
    if high == 12 and low == 11 and suited:  # AKs
        return 4

    # Strong (bucket 3): JJ-TT, AK, AQs, AJs
    if pair and high >= 8:  # TT-JJ
        return 3
    if high == 12 and low == 11:  # AKo
        return 3
    if high == 12 and low >= 9 and suited:  # AQs, AJs, ATs
        return 3

    # Medium (bucket 2): 99-77, AQ-AT, suited broadways, suited connectors 89s+
    if pair and high >= 5:  # 77-99
        return 2
    if high == 12 and low >= 8:  # ATo+
        return 2
    if suited and high >= 8 and low >= 8:  # suited broadways
        return 2
    if suited and abs(high - low) == 1 and low >= 6:  # 78s+
        return 2

    # Weak (bucket 1): 66-22, suited aces, Kxs, suited connectors
    if pair:
        return 1
    if suited and (high == 12 or high == 11):  # Axs, Kxs
        return 1
    if suited and abs(high - low) <= 2:  # suited connectors / 1-gappers
        return 1

    # Trash (bucket 0): everything else
    return 0


class OpponentModelBot:
    """
    Bayesian opponent-modeling poker bot.

    Tracks per-opponent hand-range distributions across five strength
    buckets and uses Monte-Carlo equity against the weighted ranges to
    make decisions.

    Parameters
    ----------
    simulations : int
        Number of Monte-Carlo rollouts per decision (default 400).
    """

    def __init__(self, simulations: int = 400):
        self.simulations = simulations
        # Per-opponent range distributions, keyed by player_id.
        # Each value is a numpy array of shape (NUM_BUCKETS,).
        self._ranges: dict[str, np.ndarray] = {}
        # Track which hand's history has already been consumed, to
        # avoid double-updating when act() is called multiple times
        # within the same hand.
        self._last_processed_len: int = 0
        self._current_hand_id: Optional[int] = None

    # ── public interface ──────────────────────────────────────────────────

    def act(self, state: PlayerView) -> Action:
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        opponents = state.opponents
        position = state.position
        street = state.street
        history = state.history

        # No cards → safe fallback
        if not hole:
            return self._choose("check", legal)

        # Check for a new hand (history resets or shrinks)
        if len(history) < self._last_processed_len:
            self._on_new_hand(opponents)
        self._last_processed_len = len(history)

        # Ensure every opponent has a prior
        for opp in opponents:
            if opp not in self._ranges:
                self._ranges[opp] = _uniform_prior()

        # ---- Update opponent ranges from new history entries ----
        self._update_ranges_from_history(history, state.me, board)

        # ---- Estimate equity against modelled ranges ----
        equity = self._estimate_equity_vs_ranges(
            hole, board, opponents, sims=self.simulations
        )

        # ---- Position adjustment ----
        pos_tight = self._get_position_tightness(position)
        if pos_tight > 0.5:  # early position → tighter thresholds
            bet_thresh = 0.68
            raise_thresh = 0.62
        else:
            bet_thresh = 0.58
            raise_thresh = 0.52

        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0

        # ---- Decision logic ----
        if to_call > 0:
            my_stack = state.stacks.get(state.me, 0)

            # Equity below pot odds → fold
            if equity < pot_odds:
                return self._choose("fold", legal)

            # Large call relative to stack requires stronger hand
            if my_stack > 0 and to_call > my_stack * 0.4 and equity < 0.68:
                return self._choose("fold", legal)

            # Medium → call
            if equity < raise_thresh:
                return self._choose("call", legal)

            # Strong → raise
            return self._raise(pot, legal)

        # Not facing a bet
        if equity > bet_thresh:
            return self._bet(pot, legal)

        # Medium → check
        return self._choose("check", legal)

    # ── hand lifecycle ────────────────────────────────────────────────────

    def _on_new_hand(self, opponents: list[str]):
        """Reset per-hand tracking. Ranges carry over between hands but
        are blended toward uniform so stale data decays."""
        # Decay existing ranges slightly toward uniform each hand
        uniform = _uniform_prior()
        decay = 0.85  # retention factor
        for pid in list(self._ranges):
            self._ranges[pid] = _normalise(
                decay * self._ranges[pid] + (1 - decay) * uniform
            )
        # Initialise any new opponents
        for opp in opponents:
            if opp not in self._ranges:
                self._ranges[opp] = _uniform_prior()
        self._last_processed_len = 0

    # ── Bayesian range update ────────────────────────────────────────────

    def _update_ranges_from_history(self, history: list, me: str,
                                     board: list):
        """Walk through history entries and apply likelihood updates for
        each opponent action."""
        for entry in history:
            pid = entry.get("pid")
            if pid is None or pid == me:
                continue
            if pid not in self._ranges:
                self._ranges[pid] = _uniform_prior()

            action_type = entry.get("type", "check")
            entry_street = entry.get("street", "preflop")

            if entry_street == "preflop":
                table = PREFLOP_LIKELIHOODS
            else:
                table = POSTFLOP_LIKELIHOODS

            likelihood = table.get(action_type)
            if likelihood is None:
                continue

            # Context multiplier: scale raise likelihood by size relative
            # to pot when amount information is present.
            amount = entry.get("amount")
            to_call_before = entry.get("to_call_before", 0)
            if action_type in ("bet", "raise") and amount is not None and to_call_before > 0:
                ratio = amount / max(to_call_before, 1)
                # Big raises push distribution further toward strong/premium
                size_boost = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
                if ratio > 3.0:
                    size_boost = np.array([0.5, 0.6, 0.8, 1.2, 1.5])
                elif ratio > 2.0:
                    size_boost = np.array([0.7, 0.8, 1.0, 1.1, 1.3])
                likelihood = likelihood * size_boost

            # Bayesian update: posterior ∝ prior × likelihood
            self._ranges[pid] = _normalise(self._ranges[pid] * likelihood)

    # ── Equity estimation against modelled ranges ────────────────────────

    def _estimate_equity_vs_ranges(self, hole, board, opponents,
                                    sims: int = 400) -> float:
        """Monte-Carlo equity estimate where opponent hands are sampled
        from their Bayesian range distributions rather than uniformly."""
        if not opponents:
            return 1.0

        wins = 0
        ties = 0

        for _ in range(sims):
            used = list(hole) + list(board)
            opp_hands = []
            valid = True

            for opp in opponents:
                opp_hole = self._sample_hand_from_range(
                    opp, board, used
                )
                if opp_hole is None:
                    valid = False
                    break
                opp_hands.append(opp_hole)
                used = used + list(opp_hole)

            if not valid:
                continue

            # Complete the board
            full_board = self._random_board(board, used)

            my_score = eval_hand(hole, full_board)
            opp_scores = [eval_hand(oh, full_board) for oh in opp_hands]
            best_opp = max(opp_scores)

            if my_score > best_opp:
                wins += 1
            elif my_score == best_opp:
                ties += 1

        total = wins + ties * 0.5
        return total / max(sims, 1)

    def _sample_hand_from_range(self, opp: str, board: list,
                                 used: list):
        """Sample a 2-card hand for *opp* weighted by their bucket
        distribution. Rejection-samples until a valid hand in the
        target bucket is found."""
        dist = self._ranges.get(opp, _uniform_prior())

        # Pick a target bucket according to the distribution
        bucket_idx = int(np.random.choice(NUM_BUCKETS, p=dist))

        deck = self._remaining_deck(used)
        if len(deck) < 2:
            return None

        # Attempt to find a hand that falls in the target bucket.
        # Try up to 30 attempts; fall back to any random hand.
        for _ in range(30):
            hand = random.sample(deck, 2)
            if board:
                b = _hand_bucket(hand, board)
            else:
                b = _preflop_bucket_from_hole(hand)
            if b == bucket_idx:
                return hand

        # Fallback: accept any random hand (avoids infinite loop)
        return random.sample(deck, 2)

    # ── deck / board helpers (mirror MonteCarloBot) ──────────────────────

    def _remaining_deck(self, used):
        full = [(r, s) for r in RANKS for s in SUITS]
        used_set = set(tuple(c) for c in used)
        return [c for c in full if c not in used_set]

    def _random_board(self, board, used):
        deck = self._remaining_deck(used)
        need = 5 - len(board)
        if need <= 0:
            return list(board)
        cards = random.sample(deck, min(need, len(deck)))
        return list(board) + cards

    # ── action helpers (mirror MonteCarloBot) ────────────────────────────

    def _choose(self, typ, legal):
        for a in legal:
            if a["type"] == typ:
                return Action(typ)
        for a in legal:
            if a["type"] in ("call", "check"):
                return Action(a["type"])
        return Action("fold")

    def _raise(self, pot, legal):
        for a in legal:
            if a["type"] == "raise":
                stack_cap = a["max"] * 0.30
                amt = max(a["min"], min(a["max"], pot * 0.75, stack_cap))
                return Action("raise", int(amt))
        return self._choose("call", legal)

    def _bet(self, pot, legal):
        for a in legal:
            if a["type"] == "bet":
                stack_cap = a["max"] * 0.25
                amt = max(a["min"], min(a["max"], pot * 0.5, stack_cap))
                return Action("bet", int(amt))
        return self._choose("check", legal)

    def _get_position_tightness(self, position):
        position_order = {
            "UTG": 1.0, "UTG+1": 0.9, "MP": 0.7, "LJ": 0.6,
            "HJ": 0.4, "CO": 0.2, "BTN": 0.0, "SB": 0.5, "BB": 0.7,
        }
        return position_order.get(position, 0.5)

    # ── introspection ────────────────────────────────────────────────────

    def get_opponent_range(self, pid: str) -> dict[str, float]:
        """Return the current range estimate for *pid* as a readable
        dict mapping bucket name → probability."""
        dist = self._ranges.get(pid, _uniform_prior())
        return {name: float(prob) for name, prob in zip(BUCKET_NAMES, dist)}
