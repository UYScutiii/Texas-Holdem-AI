"""
core/opponent_stats.py — Per-opponent stat tracker
--------------------------------------------------
Modeled on bots/exploitative_bot.py's tracker, but with two output
methods so Path A and Path B can both consume the same source.

Path A uses bucket() for a 4-category label in the info-set key.
Path B uses to_tensor() for a fixed-shape feature vector.

This gate provides the class only. Wiring observe_action into the
engine is a Gate 2 task per path.

Both Path A and Path B import from here. This module is read-only
after Gate 1 closes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None  # graceful degradation for non-torch environments


@dataclass
class OpponentStats:
    """Snapshot of computed statistics for one opponent."""
    vpip: float           # voluntarily put in pot, last `window` hands
    pfr: float            # preflop raise frequency
    af: float             # aggression factor (bet+raise) / call
    fold_to_cbet: float
    showdown_freq: float
    sample_size: int      # how many hands of data we have


@dataclass
class _HandRecord:
    """Internal record for a single hand's observations for one seat."""
    vpip: bool = False          # did they voluntarily put money in preflop?
    pfr: bool = False           # did they raise preflop?
    saw_cbet: bool = False      # was there a cbet on flop they faced?
    folded_to_cbet: bool = False
    went_to_showdown: bool = False
    bets_and_raises: int = 0
    calls: int = 0
    actions_seen: int = 0


class OpponentStatTracker:
    """Tracks stats for all opponents at the table. One instance per match."""

    def __init__(self, n_seats: int, window: int = 50):
        self.n_seats = n_seats
        self.window = window

        # Per-seat rolling window of hand records
        self._history: dict[int, deque] = {
            i: deque(maxlen=window) for i in range(n_seats)
        }
        # Current hand's record per seat
        self._current: dict[int, _HandRecord] = {
            i: _HandRecord() for i in range(n_seats)
        }
        # Whether we've started tracking a hand
        self._in_hand = False
        self._current_street = "preflop"

    def observe_action(self, seat_idx: int, street: str, action: str,
                       pot_before: int = 0, is_cbet: bool = False):
        """Record one observed action.

        Parameters
        ----------
        seat_idx : int
        street : str ("preflop" | "flop" | "turn" | "river")
        action : str ("fold" | "check" | "call" | "bet" | "raise" | "all_in")
        pot_before : int (pot snapshot at action time)
        is_cbet : bool (True if this is a continuation bet on flop)
        """
        if seat_idx not in self._current:
            self._current[seat_idx] = _HandRecord()

        rec = self._current[seat_idx]
        self._current_street = street

        rec.actions_seen += 1

        # VPIP (preflop only, voluntary money in)
        if street == "preflop" and not rec.vpip:
            if action in ("call", "bet", "raise", "all_in"):
                rec.vpip = True

        # PFR (preflop raise)
        if street == "preflop" and not rec.pfr:
            if action in ("raise", "all_in"):
                # All-in preflop counts as a raise for PFR
                rec.pfr = True

        # Aggression factor tracking (all streets)
        if action in ("bet", "raise", "all_in"):
            rec.bets_and_raises += 1
        elif action == "call":
            rec.calls += 1

        # Fold-to-cbet tracking
        if is_cbet and street == "flop":
            rec.saw_cbet = True
            if action == "fold":
                rec.folded_to_cbet = True

    def observe_hand_end(self, seats_to_showdown: list):
        """Call at the end of each hand.

        Parameters
        ----------
        seats_to_showdown : list of int
            Seat indices that went to showdown (not folded).
        """
        for seat_idx in range(self.n_seats):
            rec = self._current.get(seat_idx, _HandRecord())
            if seat_idx in seats_to_showdown:
                rec.went_to_showdown = True
            # Only record if we saw at least one action
            if rec.actions_seen > 0:
                self._history[seat_idx].append(rec)

        # Reset for next hand
        self._current = {i: _HandRecord() for i in range(self.n_seats)}
        self._current_street = "preflop"

    def stats_for(self, seat_idx: int) -> OpponentStats:
        """Compute current stats for a given seat from rolling window."""
        history = self._history.get(seat_idx, deque())
        n = len(history)

        if n == 0:
            return OpponentStats(
                vpip=0.0, pfr=0.0, af=0.0,
                fold_to_cbet=0.0, showdown_freq=0.0,
                sample_size=0,
            )

        vpip_count = sum(1 for rec in history if rec.vpip)
        pfr_count = sum(1 for rec in history if rec.pfr)
        total_bets = sum(rec.bets_and_raises for rec in history)
        total_calls = sum(rec.calls for rec in history)
        cbet_faced = sum(1 for rec in history if rec.saw_cbet)
        cbet_folded = sum(1 for rec in history if rec.folded_to_cbet)
        showdown_count = sum(1 for rec in history if rec.went_to_showdown)

        return OpponentStats(
            vpip=vpip_count / n,
            pfr=pfr_count / n,
            af=total_bets / max(total_calls, 1),
            fold_to_cbet=cbet_folded / max(cbet_faced, 1),
            showdown_freq=showdown_count / n,
            sample_size=n,
        )

    def bucket(self, seat_idx: int) -> str:
        """Path A's 4-bucket label in {"TP","TA","LP","LA"}.

        Boundaries: vpip < 25 -> tight; af > 1.5 -> aggressive.
        Returns "TA" (default tight-aggro) when sample_size < 5.
        """
        stats = self.stats_for(seat_idx)
        if stats.sample_size < 5:
            return "TA"

        tight = stats.vpip < 0.25
        aggressive = stats.af > 1.5

        if tight and aggressive:
            return "TA"
        elif tight and not aggressive:
            return "TP"
        elif not tight and aggressive:
            return "LA"
        else:
            return "LP"

    def to_tensor(self, seat_idx: int):
        """Path B's feature vector for one opponent.

        Fixed shape, normalized roughly to [0, 1]. Includes a sample-size
        confidence scalar so the network can downweight low-sample entries.

        Returns a torch.Tensor of shape (6,).
        Features: [vpip, pfr, af_norm, fold_to_cbet, showdown_freq, confidence]
        """
        if torch is None:
            raise ImportError("torch is required for to_tensor()")

        stats = self.stats_for(seat_idx)

        # Normalize AF to roughly [0, 1] range (cap at 5.0 -> 1.0)
        af_norm = min(stats.af / 5.0, 1.0)

        # Confidence scalar: sigmoid-like ramp from 0 to 1
        # At 5 hands: ~0.25, at 20 hands: ~0.8, at 50+ hands: ~1.0
        confidence = min(stats.sample_size / 50.0, 1.0)

        return torch.tensor([
            stats.vpip,
            stats.pfr,
            af_norm,
            stats.fold_to_cbet,
            stats.showdown_freq,
            confidence,
        ], dtype=torch.float32)
