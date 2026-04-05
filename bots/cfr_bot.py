"""
Monte Carlo Counterfactual Regret Minimisation (MCCFR) Bot
----------------------------------------------------------
A simplified MCCFR agent for No-Limit Texas Hold'em that converges
toward a Nash equilibrium strategy over time.

Key design choices:
  * Bet abstraction – four sizing buckets: 33% pot, 67% pot, pot, all-in.
  * Card abstraction – preflop hand-strength tiers (10 buckets) and
    postflop hand-strength percentile bins (10 buckets).
  * External-sampling MCCFR with regret-matching.
  * Strategy profile + cumulative regret tables persist across hands
    within a session and can be serialised to disk.
"""
from __future__ import annotations

import math
import os
import pickle
import random
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from core.bot_api import Action, PlayerView
from core.engine import eval_hand, full_deck, EVAL_HAND_MAX

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

RANKS = "23456789TJQKA"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}

# Abstract action labels (indices into strategy / regret vectors)
ABSTRACT_ACTIONS: List[str] = [
    "fold",
    "check_call",     # check when no bet, call when facing a bet
    "bet_33",         # bet / raise 33% of pot
    "bet_67",         # bet / raise 67% of pot
    "bet_100",        # bet / raise 100% of pot (pot-sized)
    "all_in",         # shove
]
NUM_ACTIONS = len(ABSTRACT_ACTIONS)

# Number of Monte Carlo rollouts for postflop hand-strength estimation
_HS_SIMS = 200

# Number of preflop buckets (hand-strength tiers)
_PREFLOP_BUCKETS = 10
# Number of postflop buckets (hand-strength percentile ranges)
_POSTFLOP_BUCKETS = 10


# ═══════════════════════════════════════════════════════════════════════════════
#  Card abstraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _preflop_bucket(hole: List[Tuple[str, str]]) -> int:
    """
    Bucket a preflop hand into one of ``_PREFLOP_BUCKETS`` tiers based on a
    simplified hand-strength heuristic (inspired by Sklansky–Malmuth groups).

    Returns an integer in [0, _PREFLOP_BUCKETS-1] where higher = stronger.
    """
    if len(hole) < 2:
        return 0

    r1 = RANK_TO_INT[hole[0][0]]
    r2 = RANK_TO_INT[hole[1][0]]
    high, low = max(r1, r2), min(r1, r2)
    suited = hole[0][1] == hole[1][1]
    pair = (r1 == r2)

    # Raw score: pairs get a big bonus, high cards contribute, suited/connected
    # hands get a small bump.
    score = high + low * 0.6
    if pair:
        score += 20 + high * 1.5
    if suited:
        score += 3
    gap = high - low
    if gap <= 2 and not pair:
        score += 2  # connector / one-gapper

    # Normalise ``score`` into [0, _PREFLOP_BUCKETS-1].  Empirical range of
    # ``score`` is ~[1.2  (2-3o), ~46  (AA)].
    max_score = 46.0
    bucket = int(score / max_score * (_PREFLOP_BUCKETS - 1))
    return max(0, min(_PREFLOP_BUCKETS - 1, bucket))


def _postflop_bucket(hole: List[Tuple[str, str]],
                     board: List[Tuple[str, str]]) -> int:
    """
    Estimate hand-strength percentile via Monte-Carlo rollout, then
    bucket into one of ``_POSTFLOP_BUCKETS`` bins.

    Returns an integer in [0, _POSTFLOP_BUCKETS-1] where higher = stronger.
    """
    if not hole or len(hole) < 2 or not board:
        return _POSTFLOP_BUCKETS // 2  # neutral bucket

    used = set(tuple(c) for c in hole) | set(tuple(c) for c in board)
    remaining = [c for c in full_deck() if c not in used]

    wins = 0
    total = 0

    for _ in range(_HS_SIMS):
        if len(remaining) < 2:
            break
        opp_hand = random.sample(remaining, 2)
        # Complete board to 5 cards if needed
        used_sim = used | set(tuple(c) for c in opp_hand)
        rest = [c for c in remaining if c not in set(tuple(c) for c in opp_hand)]
        need = 5 - len(board)
        if need > 0:
            if len(rest) < need:
                continue
            extra = random.sample(rest, need)
            full_board = list(board) + extra
        else:
            full_board = list(board)

        my_score = eval_hand(list(hole), full_board)
        opp_score = eval_hand(opp_hand, full_board)

        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            wins += 0.5
        total += 1

    if total == 0:
        return _POSTFLOP_BUCKETS // 2

    equity = wins / total
    bucket = int(equity * _POSTFLOP_BUCKETS)
    return max(0, min(_POSTFLOP_BUCKETS - 1, bucket))


def _info_set_key(street: str, bucket: int, history_key: str) -> str:
    """
    Build a compact information-set key from the street, card bucket, and
    abstracted action history.
    """
    return f"{street}:{bucket}:{history_key}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Action mapping: abstract ↔ concrete
# ═══════════════════════════════════════════════════════════════════════════════

def _legal_abstract_actions(legal: List[Dict[str, Any]],
                            pot: int) -> List[int]:
    """
    Map the engine's concrete legal actions to abstract action indices.

    Returns a list of indices into ``ABSTRACT_ACTIONS`` that are available.
    """
    types = {a["type"] for a in legal}
    result: List[int] = []

    # fold / check_call are always available when their concrete counterparts are
    if "fold" in types:
        result.append(0)  # fold
    if "check" in types or "call" in types:
        result.append(1)  # check_call

    # bet / raise sizing buckets
    has_bet_raise = "bet" in types or "raise" in types
    if has_bet_raise:
        spec = next(a for a in legal if a["type"] in ("bet", "raise"))
        lo, hi = spec["min"], spec["max"]

        # Generate the four sizing targets
        sizes = {
            2: int(pot * 0.33),   # bet_33
            3: int(pot * 0.67),   # bet_67
            4: int(pot * 1.00),   # bet_100
            5: hi,                # all_in
        }

        for idx, target in sizes.items():
            # Clamp target into [lo, hi] and accept it
            clamped = max(lo, min(hi, target))
            # Avoid duplicates when multiple targets collapse to the same value
            if idx not in result:
                result.append(idx)

    return sorted(set(result)) if result else [1]  # fallback: check/call


def _abstract_to_concrete(abstract_idx: int,
                          legal: List[Dict[str, Any]],
                          pot: int) -> Action:
    """
    Convert an abstract action index back into a concrete ``Action`` the
    engine accepts.
    """
    types = {a["type"] for a in legal}
    label = ABSTRACT_ACTIONS[abstract_idx]

    if label == "fold":
        if "fold" in types:
            return Action("fold")
        # Not allowed to fold → check/call
        return _fallback_passive(legal)

    if label == "check_call":
        if "check" in types:
            return Action("check")
        if "call" in types:
            return Action("call")
        return _fallback_passive(legal)

    # Sizing actions
    frac_map = {"bet_33": 0.33, "bet_67": 0.67, "bet_100": 1.00, "all_in": None}
    frac = frac_map.get(label)

    bet_raise = [a for a in legal if a["type"] in ("bet", "raise")]
    if not bet_raise:
        return _fallback_passive(legal)

    spec = bet_raise[0]
    lo, hi = spec["min"], spec["max"]

    if frac is None:
        # all-in
        amt = hi
    else:
        amt = int(pot * frac)

    amt = max(lo, min(hi, amt))
    return Action(spec["type"], amt)


def _fallback_passive(legal: List[Dict[str, Any]]) -> Action:
    """Fallback: check > call > fold."""
    for t in ("check", "call", "fold"):
        if any(a["type"] == t for a in legal):
            return Action(t)
    # absolute last resort
    a = legal[0]
    return Action(a["type"], a.get("min"))


# ═══════════════════════════════════════════════════════════════════════════════
#  History abstraction
# ═══════════════════════════════════════════════════════════════════════════════

def _abstract_history(history: List[Dict[str, Any]], pot: int) -> str:
    """
    Compress the engine action history into a compact string of abstract
    action labels suitable for use as an information-set key suffix.

    Each action is mapped to one of our 6 abstract action labels.
    """
    tokens: List[str] = []
    for entry in history:
        atype = entry.get("type", "")
        # Fold / check / call → direct mapping
        if atype == "fold":
            tokens.append("F")
        elif atype == "check":
            tokens.append("K")
        elif atype == "call":
            tokens.append("C")
        elif atype in ("bet", "raise"):
            amt = entry.get("amount") or 0
            if pot > 0:
                ratio = amt / pot
            else:
                ratio = 1.0
            if ratio >= 0.9:
                tokens.append("A")   # all-in / pot+
            elif ratio >= 0.8:
                tokens.append("P")   # pot-sized
            elif ratio >= 0.5:
                tokens.append("M")   # ~67%
            else:
                tokens.append("S")   # ~33%
        else:
            tokens.append("?")
    return "".join(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
#  CFR Node
# ═══════════════════════════════════════════════════════════════════════════════

class _CFRNode:
    """
    Stores cumulative regret and cumulative strategy for a single
    information set.
    """
    __slots__ = ("regret_sum", "strategy_sum")

    def __init__(self):
        self.regret_sum: List[float] = [0.0] * NUM_ACTIONS
        self.strategy_sum: List[float] = [0.0] * NUM_ACTIONS

    def get_strategy(self, legal_mask: List[int]) -> List[float]:
        """
        Regret-matching: derive current strategy from positive cumulative
        regrets, restricted to ``legal_mask`` action indices.
        """
        strategy = [0.0] * NUM_ACTIONS
        pos_sum = 0.0
        for a in legal_mask:
            val = max(0.0, self.regret_sum[a])
            strategy[a] = val
            pos_sum += val

        if pos_sum > 0:
            for a in legal_mask:
                strategy[a] /= pos_sum
        else:
            # Uniform over legal actions
            n = len(legal_mask)
            for a in legal_mask:
                strategy[a] = 1.0 / n

        return strategy

    def get_average_strategy(self, legal_mask: List[int]) -> List[float]:
        """
        Average strategy is the one that converges to Nash equilibrium.
        """
        strategy = [0.0] * NUM_ACTIONS
        total = sum(self.strategy_sum[a] for a in legal_mask)
        if total > 0:
            for a in legal_mask:
                strategy[a] = self.strategy_sum[a] / total
        else:
            n = len(legal_mask)
            for a in legal_mask:
                strategy[a] = 1.0 / n
        return strategy

    def to_dict(self) -> Dict:
        return {
            "regret_sum": list(self.regret_sum),
            "strategy_sum": list(self.strategy_sum),
        }

    @staticmethod
    def from_dict(d: Dict) -> "_CFRNode":
        node = _CFRNode()
        node.regret_sum = list(d["regret_sum"])
        node.strategy_sum = list(d["strategy_sum"])
        return node


# ═══════════════════════════════════════════════════════════════════════════════
#  CFR Bot
# ═══════════════════════════════════════════════════════════════════════════════

class CFRBot:
    """
    Monte Carlo Counterfactual Regret Minimisation (MCCFR) bot.

    Parameters
    ----------
    iterations : int
        Number of MCCFR self-play iterations to run *per decision point* to
        refine regrets before choosing an action.
    profile_path : str | None
        Path for persisting regret / strategy tables. ``None`` = in-memory only.
    use_average : bool
        If ``True`` (default), play the average strategy (Nash convergent).
        If ``False``, play the current regret-matched strategy.
    """

    def __init__(
        self,
        iterations: int = 100,
        profile_path: Optional[str] = None,
        use_average: bool = True,
    ):
        self.iterations = iterations
        self.profile_path = profile_path
        self.use_average = use_average

        # Node map: info_set_key → _CFRNode
        self._nodes: Dict[str, _CFRNode] = {}

        # Session statistics
        self._hands_played = 0
        self._total_iterations = 0

        # Attempt to load persisted profile
        if profile_path:
            self.load(profile_path)

    # ──────────────────────────────────────────────────────────────────────────
    #  Public interface: act(state) → Action
    # ──────────────────────────────────────────────────────────────────────────

    def act(self, state: PlayerView) -> Action:
        """
        Choose an action for the current game state.

        1. Compute the card bucket for the current hand + board.
        2. Run ``self.iterations`` MCCFR traversals to update regrets.
        3. Select an action from the (average) strategy profile.
        """
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        street = state.street
        history = state.history or []

        # Bail fast if we have no cards (shouldn't happen, but be safe)
        if not hole or len(hole) < 2:
            return _fallback_passive(legal)

        # ── Card abstraction ────────────────────────────────────
        if street == "preflop":
            bucket = _preflop_bucket(hole)
        else:
            bucket = _postflop_bucket(hole, board)

        # ── History abstraction ─────────────────────────────────
        hist_key = _abstract_history(history, pot)

        # ── Information-set key ─────────────────────────────────
        info_key = _info_set_key(street, bucket, hist_key)

        # ── Legal abstract actions ──────────────────────────────
        legal_mask = _legal_abstract_actions(legal, pot)

        # ── Run MCCFR iterations to refine regrets ──────────────
        self._run_iterations(info_key, legal_mask, pot, hole, board, street)

        # ── Choose action from strategy ─────────────────────────
        node = self._get_node(info_key)
        if self.use_average:
            strategy = node.get_average_strategy(legal_mask)
        else:
            strategy = node.get_strategy(legal_mask)

        # Sample from strategy distribution
        abstract_idx = self._sample_action(strategy, legal_mask)
        self._hands_played += 1

        return _abstract_to_concrete(abstract_idx, legal, pot)

    # ──────────────────────────────────────────────────────────────────────────
    #  MCCFR iteration (simplified external-sampling)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_iterations(
        self,
        info_key: str,
        legal_mask: List[int],
        pot: int,
        hole: List[Tuple[str, str]],
        board: List[Tuple[str, str]],
        street: str,
    ):
        """
        Run ``self.iterations`` simplified MCCFR traversals rooted at the
        current decision node.

        Because we don't have access to a full game-tree simulator inside the
        bot, we use a *one-step look-ahead with rollout*: for each abstract
        action, simulate the expected value via Monte-Carlo equity estimation,
        then update regrets as if each action led to a terminal node.
        """
        node = self._get_node(info_key)

        for _ in range(self.iterations):
            strategy = node.get_strategy(legal_mask)

            # Accumulate strategy for average computation
            for a in legal_mask:
                node.strategy_sum[a] += strategy[a]

            # Compute utility for each legal abstract action via rollout
            action_values = {}
            for a in legal_mask:
                action_values[a] = self._estimate_action_value(
                    a, pot, hole, board, street
                )

            # Expected value under current strategy
            ev = sum(strategy[a] * action_values.get(a, 0.0) for a in legal_mask)

            # Update regrets
            for a in legal_mask:
                regret = action_values[a] - ev
                node.regret_sum[a] += regret

        self._total_iterations += self.iterations

    def _estimate_action_value(
        self,
        abstract_idx: int,
        pot: int,
        hole: List[Tuple[str, str]],
        board: List[Tuple[str, str]],
        street: str,
    ) -> float:
        """
        Estimate the expected value of taking the given abstract action.

        Uses a simplified model:
          * fold → lose current investment (value = -1)
          * check/call → equity × pot (minus cost to call, estimated)
          * bet/raise → equity × (pot + sizing) adjusted for fold equity

        Equity is estimated via Monte-Carlo rollout.
        """
        label = ABSTRACT_ACTIONS[abstract_idx]

        # Quick equity estimate
        equity = self._quick_equity(hole, board)

        if label == "fold":
            return -1.0

        if label == "check_call":
            # Value is proportional to equity
            return equity * 2.0 - 1.0  # scaled to [-1, 1]

        # Bet / raise actions
        frac_map = {"bet_33": 0.33, "bet_67": 0.67, "bet_100": 1.00, "all_in": 2.0}
        sizing_frac = frac_map.get(label, 0.5)

        # Model fold equity: opponents fold more often against larger bets
        # Logistic curve: fold_prob increases with bet size
        fold_equity = 0.3 * min(1.0, sizing_frac)

        # When opponent calls, we see a showdown
        showdown_value = equity * 2.0 - 1.0

        # When opponent folds, we win the pot (value ≈ +1 adjusted for risk)
        fold_value = 0.5

        value = fold_equity * fold_value + (1.0 - fold_equity) * showdown_value

        # Penalise large bets slightly to avoid reckless aggression
        risk_penalty = 0.05 * sizing_frac
        value -= risk_penalty

        return value

    def _quick_equity(
        self,
        hole: List[Tuple[str, str]],
        board: List[Tuple[str, str]],
    ) -> float:
        """
        Fast equity estimate against one random opponent.
        Uses fewer simulations than the bucketing function for speed.
        """
        if not hole or len(hole) < 2:
            return 0.5

        # Preflop: use the bucket as a rough equity proxy
        if not board:
            bucket = _preflop_bucket(hole)
            return 0.3 + 0.5 * (bucket / (_PREFLOP_BUCKETS - 1))

        used = set(tuple(c) for c in hole) | set(tuple(c) for c in board)
        remaining = [c for c in full_deck() if c not in used]

        wins = 0
        total = 0
        sims = 80  # fewer sims for speed during MCCFR iterations

        for _ in range(sims):
            if len(remaining) < 2:
                break
            opp = random.sample(remaining, 2)
            rest = [c for c in remaining if c not in set(tuple(x) for x in opp)]
            need = 5 - len(board)
            if need > 0 and len(rest) < need:
                continue
            if need > 0:
                extra = random.sample(rest, need)
                fb = list(board) + extra
            else:
                fb = list(board)

            my_s = eval_hand(list(hole), fb)
            op_s = eval_hand(opp, fb)
            if my_s > op_s:
                wins += 1
            elif my_s == op_s:
                wins += 0.5
            total += 1

        return wins / total if total > 0 else 0.5

    # ──────────────────────────────────────────────────────────────────────────
    #  Node management
    # ──────────────────────────────────────────────────────────────────────────

    def _get_node(self, key: str) -> _CFRNode:
        if key not in self._nodes:
            self._nodes[key] = _CFRNode()
        return self._nodes[key]

    def _sample_action(self, strategy: List[float], legal_mask: List[int]) -> int:
        """Sample an action index from the strategy distribution."""
        r = random.random()
        cumulative = 0.0
        for a in legal_mask:
            cumulative += strategy[a]
            if r <= cumulative:
                return a
        return legal_mask[-1]  # fallback to last legal action

    # ──────────────────────────────────────────────────────────────────────────
    #  Persistence: save / load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """
        Persist the regret and strategy tables to disk as a pickle file.
        """
        path = path or self.profile_path
        if not path:
            return

        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)

        data = {
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "hands_played": self._hands_played,
            "total_iterations": self._total_iterations,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Optional[str] = None):
        """
        Load regret and strategy tables from a pickle file.
        """
        path = path or self.profile_path
        if not path or not os.path.exists(path):
            return

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._nodes = {
                k: _CFRNode.from_dict(v) for k, v in data["nodes"].items()
            }
            self._hands_played = data.get("hands_played", 0)
            self._total_iterations = data.get("total_iterations", 0)
            print(f"[CFRBot] Loaded profile from {path} "
                  f"({len(self._nodes)} info sets, "
                  f"{self._total_iterations} iterations)")
        except Exception as e:
            print(f"[CFRBot] Could not load profile from {path}: {e}")
            self._nodes = {}

    # ──────────────────────────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics about the CFR profile."""
        return {
            "info_sets": len(self._nodes),
            "hands_played": self._hands_played,
            "total_iterations": self._total_iterations,
        }
