"""
Monte Carlo Counterfactual Regret Minimisation (MCCFR) Bot
----------------------------------------------------------
A simplified MCCFR agent for No-Limit Texas Hold'em that converges
toward a Nash equilibrium strategy over time.

Key design choices:
  * Bet abstraction – six sizing buckets: 33/50/67/75/100% pot + all-in.
  * Card abstraction – preflop hand-strength tiers (10 buckets) and
    postflop hand-strength percentile bins (10 buckets).
  * Position abstraction – 4 positional buckets (early/middle/late/blinds).
  * SPR abstraction – 3 stack-to-pot ratio buckets (low/mid/high).
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
from core.engine import eval_hand, _FULL_DECK, EVAL_HAND_MAX
from core.equity import equity as _canonical_equity
from core.equity import equity_bucket as _canonical_equity_bucket
from core.action_history import ActionEvent, tokenize as _canonical_tokenize

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
    "bet_50",         # bet / raise 50% of pot (half-pot)
    "bet_67",         # bet / raise 67% of pot
    "bet_75",         # bet / raise 75% of pot (three-quarter pot)
    "bet_100",        # bet / raise 100% of pot (pot-sized)
    "all_in",         # shove
]
NUM_ACTIONS = len(ABSTRACT_ACTIONS)

# Number of Monte Carlo rollouts for postflop hand-strength estimation
_HS_SIMS = 100

# Number of preflop buckets (hand-strength tiers)
_PREFLOP_BUCKETS = 20
# Number of postflop buckets (hand-strength percentile ranges)
_POSTFLOP_BUCKETS = 20


# ═══════════════════════════════════════════════════════════════════════════════
#  Position & SPR abstraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Map engine position labels into 4 strategic buckets so CFR can learn
# different opening/calling ranges for each seat category.
_POSITION_BUCKETS = {
    # Late position: most info, widest range
    "BTN": "late", "CO": "late", "HJ": "late",
    # Middle position
    "MP": "middle", "LJ": "middle",
    # Early position: tightest range
    "UTG": "early", "UTG+1": "early", "UTG+2": "early",
    # Blinds: posted dead money, last to act preflop
    "SB": "blinds", "BB": "blinds",
}


def _position_bucket(position: str) -> str:
    """Compress engine position label into one of 4 strategic buckets."""
    return _POSITION_BUCKETS.get(position, "middle")  # safe fallback


def _spr_bucket(hero_stack: int, pot: int, opp_stacks: list) -> str:
    """
    Classify the effective stack-to-pot ratio.

    Effective stack = min(hero, biggest active opponent) — that's the
    most chips that can actually be wagered between us.
    """
    if pot <= 0:
        # Preflop before blinds posted — treat as deep
        return "high"
    effective = hero_stack
    if opp_stacks:
        effective = min(hero_stack, max(opp_stacks))
    spr = effective / pot
    if spr < 5:    return "low"     # commitment/stack-off territory
    if spr < 15:   return "mid"     # standard play
    return "high"                   # deep, implied odds matter


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
                     board: List[Tuple[str, str]],
                     n_opponents: int) -> int:
    """
    Estimate hand-strength percentile via Monte-Carlo rollout against
    ``n_opponents`` random hands, then bucket into one of
    ``_POSTFLOP_BUCKETS`` bins.

    In multiway pots hero must beat ALL opponents to "win".

    Returns an integer in [0, _POSTFLOP_BUCKETS-1] where higher = stronger.

    Delegates to core.equity.equity_bucket for the canonical implementation.
    """
    return _canonical_equity_bucket(
        hole, board, n_opponents,
        n_buckets=_POSTFLOP_BUCKETS, n_sims=_HS_SIMS,
    )


def _info_set_key(street: str, bucket: int, history_key: str,
                  n_opponents: int, position_bucket: str,
                  spr_bucket: str) -> str:
    """
    Build a compact information-set key from the street, active opponent
    count, position bucket, SPR bucket, card bucket, and abstracted
    action history.

    Including all dimensions means CFR learns separate strategies for
    different positions, stack depths, and table sizes.
    """
    return (f"{street}:{n_opponents}:{position_bucket}"
            f":{spr_bucket}:{bucket}:{history_key}")


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

        # Generate the six sizing targets (indices match ABSTRACT_ACTIONS)
        sizes = {
            2: int(pot * 0.33),   # bet_33
            3: int(pot * 0.50),   # bet_50
            4: int(pot * 0.67),   # bet_67
            5: int(pot * 0.75),   # bet_75
            6: int(pot * 1.00),   # bet_100
            7: hi,                # all_in
        }

        seen_amts = set()
        for idx, target in sizes.items():
            # Clamp target into [lo, hi], then skip duplicate concrete bets.
            clamped = max(lo, min(hi, target))
            if clamped in seen_amts:
                continue
            seen_amts.add(clamped)
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
    frac_map = {
        "bet_33": 0.33, "bet_50": 0.50, "bet_67": 0.67,
        "bet_75": 0.75, "bet_100": 1.00, "all_in": None,
    }
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

    Delegates to core.action_history.tokenize for the canonical
    implementation. Constructs ActionEvent objects from raw history
    dicts, using pot_before from the engine (falls back to current pot
    for old-format histories).

    Tokens:
      F = fold, K = check, C = call,
      S = small (~33%), Q = quarter-pot-ish (~50%),
      M = medium (~67%), L = large (~75%),
      P = pot-sized (~100%), A = all-in / over-pot
    """
    events = []
    for entry in history:
        atype = entry.get("type", "")
        amt = entry.get("amount") or 0
        ref_pot = entry.get("pot_before", pot)
        events.append(ActionEvent(
            seat=0,  # seat not used by tokenizer
            street=entry.get("street", "preflop"),
            action=atype if atype else "check",
            amount=int(amt),
            pot_before=int(ref_pot),
        ))
    return _canonical_tokenize(events)


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
        inference_mode: bool = False,
    ):
        self.iterations = iterations
        self.profile_path = profile_path
        self.use_average = use_average
        # When True, skip _run_iterations during act() so the loaded regret
        # table is used as-is without being overwritten by online updates.
        self.inference_mode = inference_mode

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
        2. If NOT in inference_mode, run MCCFR iterations to update regrets.
        3. Select an action from the (average) strategy profile, falling back
           to an equity-based heuristic for unseen information sets.
        """
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        street = state.street
        history = state.history or []
        hero_stack = int(state.stacks.get(state.me, 0))
        call_amount = max(0, int(to_call))

        # How many opponents are still active (not folded)?  The engine's
        # PlayerView.opponents already excludes folded players.
        n_opp = len(state.opponents) if state.opponents else 1
        n_opp = max(1, n_opp)  # at least 1 opponent for the math to work

        # Bail fast if we have no cards (shouldn't happen, but be safe)
        if not hole or len(hole) < 2:
            return _fallback_passive(legal)

        # ── Card abstraction ────────────────────────────────────
        if street == "preflop":
            bucket = _preflop_bucket(hole)
        else:
            bucket = _postflop_bucket(hole, board, n_opponents=n_opp)

        # ── History abstraction ─────────────────────────────────
        hist_key = _abstract_history(history, pot)

        # ── Position & SPR abstraction ──────────────────────────
        pos_b = _position_bucket(state.position)
        # Collect active opponent stacks for effective-stack calculation
        opp_stacks = [
            int(state.stacks.get(o, 0))
            for o in (state.opponents or [])
            if int(state.stacks.get(o, 0)) > 0
        ]
        spr_b = _spr_bucket(hero_stack, pot, opp_stacks)

        # ── Information-set key (includes opponent count, position, SPR) ──
        info_key = _info_set_key(street, bucket, hist_key,
                                 n_opponents=n_opp,
                                 position_bucket=pos_b,
                                 spr_bucket=spr_b)

        # ── Legal abstract actions ──────────────────────────────
        legal_mask = _legal_abstract_actions(legal, pot)

        # ── MCCFR updates: only during training, never during inference ──
        # Running iterations during live play corrupts the loaded regret
        # table with noise from the simplified value function.
        if not self.inference_mode:
            self._run_iterations(info_key, legal_mask, pot, hole, board,
                                 street, n_opponents=n_opp,
                                 call_amount=call_amount,
                                 hero_stack=hero_stack)

        # ── Choose action from strategy ─────────────────────────
        node = self._nodes.get(info_key)

        # Unseen node (especially common in multiway play with a HU-trained
        # table): fall back to an equity-based heuristic rather than
        # uniform random, which is what a blank _CFRNode would give.
        if node is None or sum(node.strategy_sum) == 0.0:
            equity = self._quick_equity(hole, board, n_opponents=n_opp)
            return self._heuristic_action(legal_mask, equity, pot, to_call, legal)

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
        n_opponents: int,
        call_amount: int,
        hero_stack: int,
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

        # Compute equity once per decision point (outside the iteration loop)
        # so that the Monte-Carlo rollout is not repeated on every iteration.
        equity = self._quick_equity(hole, board, n_opponents=n_opponents)

        for _ in range(self.iterations):
            strategy = node.get_strategy(legal_mask)

            # Accumulate strategy for average computation
            for a in legal_mask:
                node.strategy_sum[a] += strategy[a]

            # Compute utility for each legal abstract action via rollout
            action_values = {}
            for a in legal_mask:
                action_values[a] = self._estimate_action_value(
                    a,
                    pot,
                    equity,
                    n_opponents=n_opponents,
                    call_amount=call_amount,
                    hero_stack=hero_stack,
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
        equity: float,
        n_opponents: int,
        call_amount: int,
        hero_stack: int,
    ) -> float:
        """
        Estimate the chip EV of taking the given abstract action.

        Fold is the neutral reference point. Other actions are scored in
        chips, then normalized by pot size so regrets stay comparable across
        short-stack and deep-stack spots.
        """
        label = ABSTRACT_ACTIONS[abstract_idx]
        pot = max(0, int(pot))
        hero_stack = max(0, int(hero_stack))
        call_amount = max(0, int(call_amount))

        if label == "fold":
            # Neutral reference point — we stop putting chips in.
            return 0.0

        if label == "check_call":
            if call_amount == 0:
                # Free check: no extra risk, so we realize our pot share.
                ev = equity * pot
            else:
                # Calling pays only the amount we can cover.
                cost = min(call_amount, hero_stack)
                ev = equity * (pot + cost) - (1.0 - equity) * cost
            return ev / max(pot, 1)

        # Bet / raise actions
        frac_map = {
            "bet_33": 0.33, "bet_50": 0.50, "bet_67": 0.67,
            "bet_75": 0.75, "bet_100": 1.00, "all_in": 2.0,
        }
        sizing_frac = frac_map.get(label, 0.5)
        bet_size = min(sizing_frac * pot, hero_stack)

        # Bigger bets generate more folds, but with diminishing returns.
        fold_equity = min(0.45, 0.20 * sizing_frac ** 0.7)

        # If everyone folds, we win the pot that already exists.
        fold_value = pot

        # If called, villain matches the bet. We can win the larger pot,
        # but we must subtract the chips we invested.
        called_pot = pot + 2 * bet_size
        called_value = equity * called_pot - bet_size

        # Large bets are volatile; keep a simple chip-risk penalty.
        risk_penalty = 0.05 * bet_size

        ev = fold_equity * fold_value + (1.0 - fold_equity) * called_value
        ev -= risk_penalty

        return ev / max(pot, 1)

    def _quick_equity(
        self,
        hole: List[Tuple[str, str]],
        board: List[Tuple[str, str]],
        n_opponents: int,
    ) -> float:
        """
        Fast Monte-Carlo equity estimate against ``n_opponents`` random
        opponents.

        Delegates to core.equity.equity for the canonical implementation.
        Uses 100 sims for tighter MC estimates (~±0.022 noise).
        """
        return _canonical_equity(hole, board, n_opponents, n_sims=100)

    def _heuristic_action(
        self,
        legal_mask: List[int],
        equity: float,
        pot: int,
        to_call: int,
        legal: List[Dict[str, Any]],
    ) -> Action:
        """
        Equity-based fallback for information sets not seen during training.

        Tiers:
          equity ≥ 0.65  → bet (raise if possible) for value
          equity ≥ 0.45  → call/check if pot odds justify it, else fold
          equity < 0.45  → check if free, else fold
        """
        if equity >= 0.65:
            # Strong hand: bet for value using the largest available sizing.
            bet_actions = [a for a in legal_mask if a >= 2]
            if bet_actions:
                return _abstract_to_concrete(max(bet_actions), legal, pot)
            # No bet available → check/call
            return _abstract_to_concrete(1, legal, pot)

        if equity >= 0.45:
            # Marginal hand: call only if pot odds warrant it.
            total = pot + to_call
            pot_odds = to_call / total if total > 0 else 0.0
            if pot_odds <= equity:
                return _abstract_to_concrete(1, legal, pot)   # check/call
            # Bad odds → fold if allowed, else forced call
            if 0 in legal_mask:
                return _abstract_to_concrete(0, legal, pot)   # fold
            return _abstract_to_concrete(1, legal, pot)

        # Weak hand: check for free, otherwise fold.
        if to_call == 0 and 1 in legal_mask:
            return _abstract_to_concrete(1, legal, pot)       # free check
        if 0 in legal_mask:
            return _abstract_to_concrete(0, legal, pot)       # fold
        return _abstract_to_concrete(1, legal, pot)           # forced call

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

        The write is atomic: data is written to ``path + ".tmp"`` first,
        flushed and fsynced, then renamed over ``path``.  A crash or
        KeyboardInterrupt during the dump therefore cannot corrupt the
        existing checkpoint.
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
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

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
            loaded_nodes = {
                k: _CFRNode.from_dict(v) for k, v in data["nodes"].items()
            }
            # Keys must have >= 5 colons to match the current format:
            # street:n_opp:position:spr:bucket:history
            self._nodes = {
                k: node for k, node in loaded_nodes.items()
                if k.count(":") >= 5
            }
            self._hands_played = data.get("hands_played", 0)
            self._total_iterations = data.get("total_iterations", 0)
            dropped = len(loaded_nodes) - len(self._nodes)
            if dropped:
                print(
                    f"[CFRBot] Dropped {dropped} old-format keys "
                    f"(pre-position/SPR). {len(self._nodes)} valid keys remain."
                )
                if not self._nodes:
                    print(
                        "  ⚠ Profile has no valid keys for current code. Bot will fall back\n"
                        "    to heuristic until a fresh profile is trained."
                    )
            print(f"[CFRBot] Loaded profile from {path} "
                  f"({len(self._nodes)} info sets, "
                  f"{self._total_iterations} iterations)")
        except Exception as e:
            print(f"[CFRBot] Could not load profile from {path}: {e}")
            self._nodes = {}
            return

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
