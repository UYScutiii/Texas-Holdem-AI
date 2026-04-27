"""
sanity_action_history.py — covers ActionEvent + tokenize + to_tensor
--------------------------------------------------------------------
Gate 1 sanity coverage for core/action_history.py.
"""
import sys
sys.path.insert(0, ".")

from core.action_history import (
    ActionEvent, extract_history, tokenize, to_tensor, FEATURE_DIM,
)
from core.bot_api import PlayerView

PASS = True

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: extract_history from a known PlayerView
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 1: extract_history from PlayerView")
print("=" * 60)

view = PlayerView(
    me="P0",
    street="flop",
    position="BTN",
    hole_cards=[("A", "h"), ("K", "s")],
    board=[("Q", "d"), ("J", "c"), ("T", "h")],
    pot=100,
    to_call=20,
    min_raise=40,
    max_raise=500,
    legal_actions=[
        {"type": "fold"},
        {"type": "call"},
        {"type": "raise", "min": 40, "max": 500},
    ],
    stacks={"P0": 500, "P1": 300, "P2": 200},
    opponents=["P1", "P2"],
    history=[
        {"street": "preflop", "pid": "P1", "type": "raise", "amount": 20, "pot_before": 15},
        {"street": "preflop", "pid": "P0", "type": "call", "amount": None, "pot_before": 35},
        {"street": "preflop", "pid": "P2", "type": "fold", "amount": None, "pot_before": 55},
        {"street": "flop", "pid": "P1", "type": "bet", "amount": 40, "pot_before": 55},
    ],
)

events = extract_history(view)
print(f"  Extracted {len(events)} events")

# Verify event 0: P1 raises 20 with pot_before=15
e0 = events[0]
print(f"  Event 0: seat={e0.seat}, street={e0.street}, action={e0.action}, "
      f"amount={e0.amount}, pot_before={e0.pot_before}")
assert e0.action == "raise", f"Expected 'raise', got {e0.action}"
assert e0.amount == 20, f"Expected amount=20, got {e0.amount}"
assert e0.pot_before == 15, f"Expected pot_before=15, got {e0.pot_before}"

# Verify event 2: fold
e2 = events[2]
assert e2.action == "fold", f"Expected 'fold', got {e2.action}"
assert e2.amount == 0, f"Expected amount=0, got {e2.amount}"

print("  ✅ PASS — amounts match engine history directly (total, not delta)")

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: Tokenizer basic
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 2: Tokenizer basic")
print("=" * 60)

# Construct known events
known_events = [
    ActionEvent(seat=0, street="preflop", action="fold", amount=0, pot_before=30),
    ActionEvent(seat=1, street="preflop", action="check", amount=0, pot_before=30),
    ActionEvent(seat=2, street="preflop", action="call", amount=10, pot_before=30),
    ActionEvent(seat=0, street="flop", action="bet", amount=10, pot_before=50),  # 10/50 = 0.20 → S
    ActionEvent(seat=1, street="flop", action="raise", amount=25, pot_before=60),  # 25/60 = 0.417 → Q
    ActionEvent(seat=2, street="flop", action="bet", amount=40, pot_before=85),  # 40/85 = 0.47 → Q
]

tokens = tokenize(known_events)
print(f"  Token string: {tokens}")
expected = "FKCSQQ"
if tokens == expected:
    print(f"  ✅ PASS — matches expected '{expected}'")
else:
    print(f"  ❌ FAIL — expected '{expected}', got '{tokens}'")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3: Tokenizer regression — pot-sized bet stays "P"
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 3: Tokenizer regression — pot-sized bet stays P")
print("=" * 60)

# A preflop pot-sized bet stays "P" forever even after the pot grows
# This was the session bug fix: use pot_before, not current pot
evt_pot_bet = ActionEvent(seat=0, street="preflop", action="bet",
                          amount=30, pot_before=30)  # 30/30 = 1.0 → P
tok = tokenize([evt_pot_bet])
print(f"  pot-sized bet (30/30): token = '{tok}'")
if tok == "P":
    print("  ✅ PASS — pot-sized bet stays 'P'")
else:
    print(f"  ❌ FAIL — expected 'P', got '{tok}'")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4: Tokenizer ratio convention
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 4: Tokenizer ratio convention")
print("=" * 60)

# amount=80, pot_before=80 → ratio 1.0 → P
evt_p = ActionEvent(seat=0, street="flop", action="bet", amount=80, pot_before=80)
tok_p = tokenize([evt_p])
print(f"  amount=80, pot_before=80: token = '{tok_p}'")
if tok_p == "P":
    print("  ✅ PASS")
else:
    print(f"  ❌ FAIL — expected 'P', got '{tok_p}'")
    PASS = False

# amount=40, pot_before=80 → ratio 0.5 → Q
evt_q = ActionEvent(seat=0, street="flop", action="bet", amount=40, pot_before=80)
tok_q = tokenize([evt_q])
print(f"  amount=40, pot_before=80: token = '{tok_q}'")
if tok_q == "Q":
    print("  ✅ PASS")
else:
    print(f"  ❌ FAIL — expected 'Q', got '{tok_q}'")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 5: Each token is producible
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 5: All tokens producible")
print("=" * 60)

# Construct events that produce each token
all_events = [
    ActionEvent(seat=0, street="preflop", action="fold", amount=0, pot_before=100),      # F
    ActionEvent(seat=0, street="preflop", action="check", amount=0, pot_before=100),     # K
    ActionEvent(seat=0, street="preflop", action="call", amount=10, pot_before=100),     # C
    ActionEvent(seat=0, street="flop", action="bet", amount=20, pot_before=100),         # S (0.20)
    ActionEvent(seat=0, street="flop", action="bet", amount=50, pot_before=100),         # Q (0.50)
    ActionEvent(seat=0, street="flop", action="bet", amount=67, pot_before=100),         # M (0.67)
    ActionEvent(seat=0, street="flop", action="bet", amount=75, pot_before=100),         # L (0.75)
    ActionEvent(seat=0, street="flop", action="bet", amount=100, pot_before=100),        # P (1.00)
    ActionEvent(seat=0, street="flop", action="bet", amount=150, pot_before=100),        # A (1.50)
]

all_tokens = tokenize(all_events)
print(f"  All tokens: '{all_tokens}'")
expected_all = "FKCSQMLPA"
if all_tokens == expected_all:
    print(f"  ✅ PASS — all tokens match '{expected_all}'")
else:
    print(f"  ❌ FAIL — expected '{expected_all}', got '{all_tokens}'")
    PASS = False

# Verify each individual token is present
for tok_char in "FKCSQMLPA":
    if tok_char in all_tokens:
        print(f"  '{tok_char}' present ✅")
    else:
        print(f"  '{tok_char}' MISSING ❌")
        PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 6: Tensor shape stability
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 6: Tensor output")
print("=" * 60)

try:
    import torch

    # Short sequence
    short_events = [
        ActionEvent(seat=0, street="preflop", action="raise", amount=20, pot_before=15),
        ActionEvent(seat=1, street="preflop", action="call", amount=10, pot_before=35),
    ]
    t_short = to_tensor(short_events, max_len=64)
    print(f"  Short sequence ({len(short_events)} events) → shape: {t_short.shape}")
    assert t_short.shape == (64, FEATURE_DIM), f"Wrong shape: {t_short.shape}"

    # Empty sequence
    t_empty = to_tensor([], max_len=64)
    print(f"  Empty sequence → shape: {t_empty.shape}")
    assert t_empty.shape == (64, FEATURE_DIM), f"Wrong shape: {t_empty.shape}"

    # Shapes match
    if t_short.shape == t_empty.shape:
        print(f"  ✅ PASS — stable shape (max_len=64, feature_dim={FEATURE_DIM})")
    else:
        print("  ❌ FAIL — shapes differ")
        PASS = False

    # Padding mask is zero where padded
    # The mask is the last channel (index -1) of each row
    mask_idx = FEATURE_DIM - 1
    # First 2 rows should have mask=1, rest should have mask=0
    if t_short[0, mask_idx] == 1.0 and t_short[1, mask_idx] == 1.0:
        print("  ✅ PASS — mask=1 for real events")
    else:
        print("  ❌ FAIL — mask not set for real events")
        PASS = False

    if t_short[2, mask_idx] == 0.0 and t_short[63, mask_idx] == 0.0:
        print("  ✅ PASS — mask=0 for padded events")
    else:
        print(f"  ❌ FAIL — padding mask not 0 (idx2={t_short[2, mask_idx]}, idx63={t_short[63, mask_idx]})")
        PASS = False

    # Empty tensor should have all zeros
    if t_empty.sum() == 0.0:
        print("  ✅ PASS — empty tensor is all zeros")
    else:
        print("  ❌ FAIL — empty tensor has non-zero values")
        PASS = False

    # Round-trip spot-check: known events → tensor → check features
    # Event 0 is seat=0, street=preflop, action=raise
    # Seat one-hot: idx 0 should be 1
    assert t_short[0, 0] == 1.0, f"Seat one-hot wrong: {t_short[0, 0]}"
    # Street one-hot: preflop is idx 0, offset at position 10 (after seats)
    assert t_short[0, 10] == 1.0, f"Street one-hot wrong: {t_short[0, 10]}"
    # Action one-hot: raise is idx 4, offset at position 14 (after 10 seats + 4 streets)
    assert t_short[0, 14 + 4] == 1.0, f"Action one-hot wrong: {t_short[0, 18]}"
    print("  ✅ PASS — round-trip spot-check passed")

    # No NaN
    if not torch.isnan(t_short).any():
        print("  ✅ PASS — no NaN in tensor")
    else:
        print("  ❌ FAIL — NaN detected")
        PASS = False

except ImportError:
    print("  ⚠ SKIP — torch not available")

print()

# ═══════════════════════════════════════════════════════════════════════════
#  OVERALL
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
if PASS:
    print("ALL CHECKS PASSED ✅")
else:
    print("SOME CHECKS FAILED ❌")
    sys.exit(1)
print("=" * 60)
