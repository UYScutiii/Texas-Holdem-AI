"""
sanity_opponent_stats.py — covers OpponentStatTracker
------------------------------------------------------
Gate 1 sanity coverage for core/opponent_stats.py.
"""
import sys
sys.path.insert(0, ".")

from core.opponent_stats import OpponentStats, OpponentStatTracker

PASS = True

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: Synthetic 10x fold preflop → VPIP near 0
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 1: 10x fold preflop → VPIP near 0")
print("=" * 60)

tracker = OpponentStatTracker(n_seats=6, window=50)

# Seat 2 folds preflop 10 times
for _ in range(10):
    tracker.observe_action(2, "preflop", "fold")
    tracker.observe_hand_end([])  # no one goes to showdown

stats = tracker.stats_for(2)
print(f"  Seat 2 VPIP after 10 folds: {stats.vpip:.4f}")
print(f"  Sample size: {stats.sample_size}")

if stats.vpip < 0.05:
    print("  ✅ PASS — VPIP near 0")
else:
    print(f"  ❌ FAIL — VPIP = {stats.vpip:.4f}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: 10x raise preflop → VPIP near 1.0, AF high
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 2: 10x raise preflop → VPIP near 1.0, AF high")
print("=" * 60)

tracker2 = OpponentStatTracker(n_seats=6, window=50)

# Seat 3 raises preflop 10 times, also bets flop
for _ in range(10):
    tracker2.observe_action(3, "preflop", "raise", pot_before=30)
    tracker2.observe_action(3, "flop", "bet", pot_before=60)
    tracker2.observe_hand_end([3])  # goes to showdown

stats2 = tracker2.stats_for(3)
print(f"  Seat 3 VPIP: {stats2.vpip:.4f}")
print(f"  Seat 3 AF: {stats2.af:.4f}")
print(f"  Seat 3 PFR: {stats2.pfr:.4f}")
print(f"  Sample size: {stats2.sample_size}")

if stats2.vpip >= 0.9:
    print("  ✅ PASS — VPIP near 1.0")
else:
    print(f"  ❌ FAIL — VPIP = {stats2.vpip:.4f}")
    PASS = False

if stats2.af >= 2.0:
    print("  ✅ PASS — AF high (aggressive)")
else:
    print(f"  ❌ FAIL — AF = {stats2.af:.4f}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3: Rolling window behavior
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 3: Rolling window")
print("=" * 60)

# Use a window of 10 for easy testing
tracker3 = OpponentStatTracker(n_seats=6, window=10)

# First 10 hands: all raises (VPIP should be 1.0)
for _ in range(10):
    tracker3.observe_action(0, "preflop", "raise", pot_before=20)
    tracker3.observe_hand_end([0])

stats_before = tracker3.stats_for(0)
print(f"  After 10 raises: VPIP = {stats_before.vpip:.4f}")

# Next 5 hands: all folds
for _ in range(5):
    tracker3.observe_action(0, "preflop", "fold")
    tracker3.observe_hand_end([])

stats_after = tracker3.stats_for(0)
print(f"  After +5 folds: VPIP = {stats_after.vpip:.4f}")

# VPIP should reflect a mix since window is 10 and we have 15 total
# The window keeps the last 10, so 5 raises + 5 folds = VPIP ~0.5
if stats_after.vpip < stats_before.vpip:
    print("  ✅ PASS — VPIP decreased after folds")
else:
    print("  ❌ FAIL — VPIP didn't decrease")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4: Bucket assignment
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 4: Bucket assignment")
print("=" * 60)

# TP: tight passive (vpip < 0.25, af <= 1.5)
tp_tracker = OpponentStatTracker(n_seats=6, window=50)
for i in range(20):
    if i < 4:  # 4/20 = 0.20 VPIP
        tp_tracker.observe_action(0, "preflop", "call", pot_before=20)
        tp_tracker.observe_action(0, "flop", "call", pot_before=40)
    else:
        tp_tracker.observe_action(0, "preflop", "fold")
    tp_tracker.observe_hand_end([0] if i < 4 else [])

tp_stats = tp_tracker.stats_for(0)
tp_bucket = tp_tracker.bucket(0)
print(f"  TP profile: VPIP={tp_stats.vpip:.2f}, AF={tp_stats.af:.2f} → bucket={tp_bucket}")
if tp_bucket == "TP":
    print("  ✅ PASS")
else:
    print(f"  ❌ FAIL — expected TP, got {tp_bucket}")
    PASS = False

# TA: tight aggressive (vpip < 0.25, af > 1.5)
ta_tracker = OpponentStatTracker(n_seats=6, window=50)
for i in range(20):
    if i < 4:  # 4/20 = 0.20 VPIP
        ta_tracker.observe_action(0, "preflop", "raise", pot_before=20)
        ta_tracker.observe_action(0, "flop", "bet", pot_before=40)
        ta_tracker.observe_action(0, "turn", "bet", pot_before=80)
    else:
        ta_tracker.observe_action(0, "preflop", "fold")
    ta_tracker.observe_hand_end([0] if i < 4 else [])

ta_stats = ta_tracker.stats_for(0)
ta_bucket = ta_tracker.bucket(0)
print(f"  TA profile: VPIP={ta_stats.vpip:.2f}, AF={ta_stats.af:.2f} → bucket={ta_bucket}")
if ta_bucket == "TA":
    print("  ✅ PASS")
else:
    print(f"  ❌ FAIL — expected TA, got {ta_bucket}")
    PASS = False

# LP: loose passive (vpip >= 0.25, af <= 1.5)
lp_tracker = OpponentStatTracker(n_seats=6, window=50)
for i in range(20):
    if i < 15:  # 15/20 = 0.75 VPIP
        lp_tracker.observe_action(0, "preflop", "call", pot_before=20)
        lp_tracker.observe_action(0, "flop", "call", pot_before=40)
    else:
        lp_tracker.observe_action(0, "preflop", "fold")
    lp_tracker.observe_hand_end([0] if i < 15 else [])

lp_stats = lp_tracker.stats_for(0)
lp_bucket = lp_tracker.bucket(0)
print(f"  LP profile: VPIP={lp_stats.vpip:.2f}, AF={lp_stats.af:.2f} → bucket={lp_bucket}")
if lp_bucket == "LP":
    print("  ✅ PASS")
else:
    print(f"  ❌ FAIL — expected LP, got {lp_bucket}")
    PASS = False

# LA: loose aggressive (vpip >= 0.25, af > 1.5)
la_tracker = OpponentStatTracker(n_seats=6, window=50)
for i in range(20):
    if i < 15:  # 15/20 = 0.75 VPIP
        la_tracker.observe_action(0, "preflop", "raise", pot_before=20)
        la_tracker.observe_action(0, "flop", "bet", pot_before=40)
        la_tracker.observe_action(0, "turn", "bet", pot_before=80)
    else:
        la_tracker.observe_action(0, "preflop", "fold")
    la_tracker.observe_hand_end([0] if i < 15 else [])

la_stats = la_tracker.stats_for(0)
la_bucket = la_tracker.bucket(0)
print(f"  LA profile: VPIP={la_stats.vpip:.2f}, AF={la_stats.af:.2f} → bucket={la_bucket}")
if la_bucket == "LA":
    print("  ✅ PASS")
else:
    print(f"  ❌ FAIL — expected LA, got {la_bucket}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 5: Default behavior — sample_size < 5 → bucket returns "TA"
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 5: Default bucket for low sample size")
print("=" * 60)

default_tracker = OpponentStatTracker(n_seats=6, window=50)
# Only 3 hands
for _ in range(3):
    default_tracker.observe_action(1, "preflop", "call", pot_before=20)
    default_tracker.observe_hand_end([1])

bucket_low = default_tracker.bucket(1)
stats_low = default_tracker.stats_for(1)
print(f"  Sample size: {stats_low.sample_size}, bucket: {bucket_low}")
if bucket_low == "TA":
    print("  ✅ PASS — default TA for low sample")
else:
    print(f"  ❌ FAIL — expected TA, got {bucket_low}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════
#  TEST 6: Tensor output
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 6: Tensor output")
print("=" * 60)

try:
    import torch

    tensor_tracker = OpponentStatTracker(n_seats=6, window=50)

    # Build some history for seat 0
    for _ in range(10):
        tensor_tracker.observe_action(0, "preflop", "raise", pot_before=20)
        tensor_tracker.observe_action(0, "flop", "bet", pot_before=60)
        tensor_tracker.observe_hand_end([0])

    t0 = tensor_tracker.to_tensor(0)
    print(f"  Seat 0 tensor shape: {t0.shape}")
    print(f"  Seat 0 tensor values: {t0.tolist()}")

    # Stable shape across seats
    t1 = tensor_tracker.to_tensor(1)  # empty seat
    print(f"  Seat 1 (empty) tensor shape: {t1.shape}")
    if t0.shape == t1.shape:
        print("  ✅ PASS — stable shape across seats")
    else:
        print("  ❌ FAIL — different shapes")
        PASS = False

    # No NaN
    if not torch.isnan(t0).any() and not torch.isnan(t1).any():
        print("  ✅ PASS — no NaN values")
    else:
        print("  ❌ FAIL — NaN detected")
        PASS = False

    # Confidence scalar increases with sample size
    conf_0 = t0[-1].item()  # last element is confidence
    conf_1 = t1[-1].item()
    print(f"  Confidence: seat 0 = {conf_0:.4f}, seat 1 = {conf_1:.4f}")
    if conf_0 > conf_1:
        print("  ✅ PASS — confidence increases with sample size")
    else:
        print("  ❌ FAIL — confidence not increasing")
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
