"""
core/action_history.py — Canonical action representation + tokenizer + tensor encoder
-------------------------------------------------------------------------------------
Two consumers (Path A wants tokens, Path B wants tensors), one source of truth.

Extracted from the inline tokenizer in bots/cfr_bot.py (session 2026-04-26).

Both Path A and Path B import from here. This module is read-only
after Gate 1 closes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None  # graceful degradation for non-torch environments


@dataclass(frozen=True)
class ActionEvent:
    """Canonical representation of a single action in the hand history."""
    seat: int
    street: str        # "preflop" | "flop" | "turn" | "river"
    action: str        # "fold" | "check" | "call" | "bet" | "raise" | "all_in"
    amount: int        # chips put in by this action.
                       # For bet/raise: TOTAL bet size (target), matching the
                       #   engine's history entry["amount"]. NOT the delta
                       #   beyond an existing bet.
                       # For call: chips matched to the existing bet.
                       # For all_in: same total convention as bet/raise.
                       # For fold/check: 0.
    pot_before: int    # snapshot at action time -- last session's bug fix


# ─── Street and action vocabularies ──────────────────────────────────────────

_STREETS = ("preflop", "flop", "turn", "river")
_STREET_TO_IDX = {s: i for i, s in enumerate(_STREETS)}

_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
_ACTION_TO_IDX = {a: i for i, a in enumerate(_ACTIONS)}


def extract_history(player_view) -> list:
    """Pull the canonical action history off the engine's PlayerView.

    `amount` matches engine history entries directly (total, not delta).

    Parameters
    ----------
    player_view : PlayerView
        The engine's read-only game state.

    Returns
    -------
    list[ActionEvent]
    """
    history = getattr(player_view, 'history', None) or []
    events = []

    # Build a mapping of player_id -> seat_idx from stacks
    stacks = getattr(player_view, 'stacks', {})
    pid_to_seat = {}
    if stacks:
        for i, pid in enumerate(stacks.keys()):
            pid_to_seat[pid] = i

    for entry in history:
        if not isinstance(entry, dict):
            continue

        pid = entry.get("pid", "")
        seat = pid_to_seat.get(pid, 0)
        street = entry.get("street", "preflop")
        action_type = entry.get("type", "check")

        # Normalize action type
        if action_type in ("bet", "raise"):
            action = action_type
        elif action_type == "fold":
            action = "fold"
        elif action_type == "check":
            action = "check"
        elif action_type == "call":
            action = "call"
        else:
            action = "check"  # safe fallback

        amount = entry.get("amount") or 0
        pot_before = entry.get("pot_before", 0)

        events.append(ActionEvent(
            seat=seat,
            street=street,
            action=action,
            amount=int(amount),
            pot_before=int(pot_before),
        ))

    return events


def tokenize(events) -> str:
    """Path A's tokenizer -- replaces the inline scheme in cfr_bot.py.

    Tokens: F (fold), K (check), C (call), S (~33%), Q (~50%), M (~67%),
            L (~75%), P (~100%), A (all-in).

    Pot-fraction ratio: ratio = event.amount / event.pot_before.
    This matches cfr_bot.py's existing convention (`amt / pot_before`)
    and the engine's amount semantics (total, not delta). Path of least
    surprise -- do NOT switch to (amount - existing_bet) / pot_before.

    Parameters
    ----------
    events : list[ActionEvent]

    Returns
    -------
    str: concatenated token string
    """
    tokens = []
    for event in events:
        action = event.action

        if action == "fold":
            tokens.append("F")
        elif action == "check":
            tokens.append("K")
        elif action == "call":
            tokens.append("C")
        elif action in ("bet", "raise", "all_in"):
            amt = event.amount
            pot = event.pot_before

            if pot > 0:
                ratio = amt / pot
            else:
                ratio = 1.0

            # Bin ratio into one of 6 sizing tokens
            if ratio >= 1.2:
                tokens.append("A")   # all-in / over-pot
            elif ratio >= 0.85:
                tokens.append("P")   # pot-sized (~100%)
            elif ratio >= 0.70:
                tokens.append("L")   # large (~75%)
            elif ratio >= 0.55:
                tokens.append("M")   # medium (~67%)
            elif ratio >= 0.40:
                tokens.append("Q")   # quarter-ish (~50%)
            else:
                tokens.append("S")   # small (~33%)
        else:
            tokens.append("?")

    return "".join(tokens)


# ─── Tensor encoder for Path B ───────────────────────────────────────────────

# Feature dimensions for tensor encoding
_N_SEATS_MAX = 10      # max seats at a table
_N_STREETS = len(_STREETS)
_N_ACTIONS = len(_ACTIONS)
# Per-event: seat_onehot + street_onehot + action_onehot + amount_norm + pot_norm + mask
FEATURE_DIM = _N_SEATS_MAX + _N_STREETS + _N_ACTIONS + 2 + 1  # = 23


def to_tensor(events, max_len=64):
    """Path B's input -- fixed-shape padded tensor for GRU consumption.

    Each event encoded as [seat_onehot, street_onehot, action_onehot,
    amount_norm, pot_norm, mask]. Shape: (max_len, feature_dim).
    Padding token has its mask channel set to 0.

    Parameters
    ----------
    events : list[ActionEvent]
    max_len : int

    Returns
    -------
    torch.Tensor of shape (max_len, FEATURE_DIM)
    """
    if torch is None:
        raise ImportError("torch is required for to_tensor()")

    tensor = torch.zeros(max_len, FEATURE_DIM, dtype=torch.float32)

    for i, event in enumerate(events[:max_len]):
        offset = 0

        # Seat one-hot (up to _N_SEATS_MAX)
        seat_idx = min(event.seat, _N_SEATS_MAX - 1)
        tensor[i, offset + seat_idx] = 1.0
        offset += _N_SEATS_MAX

        # Street one-hot
        street_idx = _STREET_TO_IDX.get(event.street, 0)
        tensor[i, offset + street_idx] = 1.0
        offset += _N_STREETS

        # Action one-hot
        action_idx = _ACTION_TO_IDX.get(event.action, 1)  # default to check
        tensor[i, offset + action_idx] = 1.0
        offset += _N_ACTIONS

        # Normalized amount (log scale, capped)
        # Use log(1 + amount) / log(1 + 10000) to normalize to ~[0, 1]
        import math
        amount_norm = math.log1p(event.amount) / math.log1p(10000)
        tensor[i, offset] = min(amount_norm, 1.0)
        offset += 1

        # Normalized pot
        pot_norm = math.log1p(event.pot_before) / math.log1p(10000)
        tensor[i, offset] = min(pot_norm, 1.0)
        offset += 1

        # Mask (1 = real event, 0 = padding)
        tensor[i, offset] = 1.0

    return tensor
