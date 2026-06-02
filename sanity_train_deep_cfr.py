"""
sanity_train_deep_cfr.py — Gate 3B sanity: training pipeline verification
--------------------------------------------------------------------------
Sections:
  1. _cfr_recurse with target collection populates buffers
  2. _cfr_recurse with buffers=None doesn't populate buffers
  3. Single train_step decreases loss
  4. 100-iteration smoke run on small variant completes under 5 min
  4b. Doc smoke trains; default short run warns on zero regret steps
  5. Checkpoint save/load round-trip
  6. Network state changes after training (anti-stub gate)
  7. AIVAT sims is configurable
  8. AIVAT leaf recoverable errors are logged
  9. Big blind inference from history
 10. All-in curriculum schedule
 11. Shadow all-in targets EXCLUDED (Key Change #2 — reversed contract)
 12. All-in detox isolation
 13. Checkpoint promotion
 14. Accelerated all-in curriculum smoke
 15. Tournament inference guardrails
 16. All existing sanity scripts still pass
"""
import io
import math
import os
import random
import subprocess
import sys
import tempfile
import time as _time
from contextlib import redirect_stdout
from copy import deepcopy

sys.path.insert(0, ".")

import torch

import training.train_deep_cfr as train_deep_cfr_module
from bots.deep_cfr_bot import (
    DeepCFRBot, DeepCFRConfig, DeepCFRNetwork,
    ReservoirBuffer, _DeepCFRGameState, NUM_ACTIONS, ABSTRACT_ACTIONS,
    _infer_big_blind,
)
from training.train_deep_cfr import (
    build_initial_state, train_step, save_checkpoint,
    load_checkpoint, pick_device, run_training, parse_args,
    all_in_policy_probability_for_iteration,
    all_in_phase_for_iteration,
    classify_canary,
    classify_extra_canary_metrics,
    decide_canary_status,
    _worst_canary_status,
    safe_checkpoint_path,
    save_promoted_checkpoint,
    warn_checkpoint_path,
)
from core.bot_api import PlayerView

PASS = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 1 — _cfr_recurse with target collection populates buffers
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 1: _cfr_recurse with target collection populates buffers")
print("=" * 60)

try:
    with redirect_stdout(io.StringIO()):
        bot1 = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                          aivat_sims=50)
    r_buf = ReservoirBuffer(capacity=10_000)
    v_buf = ReservoirBuffer(capacity=10_000)
    s_buf = ReservoirBuffer(capacity=10_000)

    random.seed(42)
    state = build_initial_state(n_seats=6, hero_seat=0)
    bot1._cfr_recurse(
        state, hero_seat=0, depth=4,
        iteration=1,
        regret_buf=r_buf, value_buf=v_buf, sizing_buf=s_buf,
    )

    print(f"  regret_buf: {len(r_buf)}")
    print(f"  value_buf:  {len(v_buf)}")
    print(f"  sizing_buf: {len(s_buf)}")

    if len(r_buf) > 0:
        print("  [PASS] — regret_buf populated")
    else:
        print("  [FAIL] — regret_buf empty")
        PASS = False

    if len(v_buf) > 0:
        print("  [PASS] — value_buf populated")
    else:
        print("  [FAIL] — value_buf empty")
        PASS = False

    if len(s_buf) >= 0:
        print(f"  [PASS] — sizing_buf has {len(s_buf)} entries (≥0 expected)")
    else:
        print("  [FAIL] — sizing_buf negative (impossible)")
        PASS = False

    # Verify entry shapes
    r_entry = r_buf.buffer[0]
    if (isinstance(r_entry, tuple) and len(r_entry) == 4 and
            isinstance(r_entry[0], dict) and
            isinstance(r_entry[1], torch.Tensor) and r_entry[1].shape == (NUM_ACTIONS,) and
            isinstance(r_entry[2], torch.Tensor) and r_entry[2].shape == (NUM_ACTIONS,) and
            isinstance(r_entry[3], float)):
        print("  [PASS] — regret entry shape: (dict, tensor[8], tensor[8], float)")
    else:
        print(f"  [FAIL] — regret entry shape wrong: len={len(r_entry)}")
        PASS = False

    v_entry = v_buf.buffer[0]
    if (isinstance(v_entry, tuple) and len(v_entry) == 2 and
            isinstance(v_entry[0], dict) and isinstance(v_entry[1], float)):
        print("  [PASS] — value entry shape: (dict, float)")
    else:
        print(f"  [FAIL] — value entry shape wrong")
        PASS = False

    if len(s_buf) > 0:
        s_entry = s_buf.buffer[0]
        if (isinstance(s_entry, tuple) and len(s_entry) == 2 and
                isinstance(s_entry[0], dict) and isinstance(s_entry[1], float)):
            print("  [PASS] — sizing entry shape: (dict, float)")
        else:
            print(f"  [FAIL] — sizing entry shape wrong")
            PASS = False

except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 2 — _cfr_recurse with buffers=None doesn't populate buffers
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 2: _cfr_recurse with buffers=None (regression test)")
print("=" * 60)

try:
    with redirect_stdout(io.StringIO()):
        bot2 = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                          aivat_sims=50)
    r_buf2 = ReservoirBuffer(capacity=10_000)
    v_buf2 = ReservoirBuffer(capacity=10_000)
    s_buf2 = ReservoirBuffer(capacity=10_000)

    random.seed(42)
    state2 = build_initial_state(n_seats=6, hero_seat=0)

    # Call WITHOUT buffer kwargs
    bot2._cfr_recurse(state2, hero_seat=0, depth=4)

    if len(r_buf2) == 0 and len(v_buf2) == 0 and len(s_buf2) == 0:
        print("  [PASS] — buffers remain empty without kwargs")
    else:
        print(f"  [FAIL] — buffers populated: r={len(r_buf2)}, v={len(v_buf2)}, s={len(s_buf2)}")
        PASS = False

except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 3 — Single train_step decreases loss
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 3: train_step decreases loss on controlled batch")
print("=" * 60)

try:
    import random as _random_mod
    import numpy as _np

    _random_mod.seed(42)
    _np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    with redirect_stdout(io.StringIO()):
        bot3 = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                          aivat_sims=50)
    device = torch.device("cpu")
    bot3.network.to(device)
    optimizer3 = torch.optim.Adam(bot3.network.parameters(), lr=1e-4)

    r3 = ReservoirBuffer(capacity=100_000)
    v3 = ReservoirBuffer(capacity=100_000)
    s3 = ReservoirBuffer(capacity=100_000)

    # Fill buffers with 50 traversals
    for i in range(50):
        random.seed(i * 7 + 1)
        state = build_initial_state(n_seats=6, hero_seat=i % 6)
        bot3.network.eval()
        bot3._cfr_recurse(
            state, hero_seat=i % 6, depth=2,
            iteration=i + 1,
            regret_buf=r3, value_buf=v3, sizing_buf=s3,
        )

    print(f"  After 50 traversals: r={len(r3)}, v={len(v3)}, s={len(s3)}")

    batch_size = min(64, len(r3))

    # Take initial loss snapshot
    bot3.network.train()
    r0, v0, s0, _ = train_step(
        bot3.network, optimizer3, r3, v3, s3, batch_size, device)
    print(f"  Initial losses:  r={r0:.6f}, v={v0:.6f}, s={s0:.6f}")

    # Run 5 more gradient steps
    for _ in range(5):
        train_step(bot3.network, optimizer3, r3, v3, s3, batch_size, device)

    # Take final loss snapshot
    r1, v1, s1, _ = train_step(
        bot3.network, optimizer3, r3, v3, s3, batch_size, device)
    print(f"  After 6 steps:   r={r1:.6f}, v={v1:.6f}, s={s1:.6f}")

    decreased = 0
    for name, before, after in [("regret", r0, r1), ("value", v0, v1), ("sizing", s0, s1)]:
        if after < before:
            decreased += 1
            print(f"  {name}: {before:.6f} → {after:.6f} (decreased [PASS])")
        else:
            print(f"  {name}: {before:.6f} → {after:.6f} (not decreased)")

    before_total = r0 + v0 + s0
    after_total = r1 + v1 + s1
    print(f"  total: {before_total:.6f} → {after_total:.6f}")

    if decreased >= 1 or after_total < before_total:
        print(f"  [PASS] — train_step improved {decreased}/3 heads "
              "or decreased total loss")
    else:
        print(f"  [FAIL] — {decreased}/3 losses decreased and total loss did not")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4 — 100-iteration smoke run on small variant
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 4: 100-iteration smoke run (small, aivat_sims=50, ≤300s)")
print("=" * 60)

smoke_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        smoke_path = f.name

    t0 = _time.monotonic()

    with redirect_stdout(io.StringIO()):
        smoke_args = parse_args([
            "--variant", "small",
            "--iterations", "100",
            "--update-interval", "10",
            "--checkpoint-interval", "50",
            "--batch-size", "32",
            "--aivat-sims", "50",
            "--save-path", smoke_path,
            "--device", "cpu",
            "--disable-collapse-canary",
        ])
        result = run_training(smoke_args)

    elapsed = _time.monotonic() - t0
    print(f"  Wall-clock: {elapsed:.1f}s")

    if elapsed <= 300:
        print("  [PASS] — completed under 300s")
    else:
        print(f"  [FAIL] — took {elapsed:.1f}s (>300s)")
        PASS = False

    # Buffer fill check
    r_len = len(result["regret_buf"])
    print(f"  regret_buf entries: {r_len}")
    if r_len >= 50:
        print("  [PASS] — regret_buf ≥ 50 entries")
    else:
        print(f"  [FAIL] — regret_buf has {r_len} entries (<50)")
        PASS = False

    # Gradient step count
    g_steps = result["gradient_steps_taken"]
    print(f"  Gradient steps: {g_steps}")
    if g_steps >= 4:
        print("  [PASS] — ≥ 4 gradient steps")
    else:
        print(f"  [FAIL] — only {g_steps} gradient steps (<4)")
        PASS = False

    # Loss stability
    r_losses = result["losses"]["regret"]
    all_losses = (
        result["losses"]["regret"] +
        result["losses"]["value"] +
        result["losses"]["sizing"]
    )
    if all(math.isfinite(x) for x in all_losses):
        print("  [PASS] — no NaN/Inf in smoke losses")
    else:
        print("  [FAIL] — NaN/Inf found in smoke losses")
        PASS = False

    if r_losses and len(r_losses) >= 2:
        initial_r = r_losses[0]
        final_r = r_losses[-1]
        print(f"  Regret loss: initial={initial_r:.6f}, final={final_r:.6f}")
        if final_r < 100.0:
            print("  [PASS] — final regret loss < 100")
        else:
            print(f"  [FAIL] — final regret loss {final_r:.6f} >= 100")
            PASS = False
    else:
        print(f"  [FAIL] — not enough regret loss history to check ({len(r_losses)} entries)")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4b — Doc smoke command and default short-run warning
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 4b: doc smoke trains; default short run warns")
print("=" * 60)

doc_path = None
warn_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        doc_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        warn_path = f.name

    with redirect_stdout(io.StringIO()) as doc_out:
        doc_args = parse_args([
            "--variant", "small",
            "--iterations", "50",
            "--update-interval", "10",
            "--checkpoint-interval", "25",
            "--batch-size", "32",
            "--aivat-sims", "50",
            "--save-path", doc_path,
            "--device", "cpu",
            "--disable-collapse-canary",
        ])
        doc_result = run_training(doc_args)
    doc_text = doc_out.getvalue()
    doc_grad_steps = doc_result["gradient_steps_taken"]
    doc_final_regret = (
        doc_result["losses"]["regret"][-1]
        if doc_result["losses"]["regret"] else float("inf")
    )
    print(f"  Doc command gradient steps: {doc_grad_steps}")
    print(f"  Doc command final regret loss: {doc_final_regret:.6f}")
    if doc_grad_steps > 0 and doc_final_regret < 100.0 and os.path.exists(doc_path):
        print("  [PASS] — doc smoke trains and saves a checkpoint")
    else:
        print("  [FAIL] — doc smoke did not produce meaningful training")
        PASS = False
    if "[WARN] Training completed with 0 gradient steps" not in doc_text:
        print("  [PASS] — doc smoke emitted no zero-step warning")
    else:
        print("  [FAIL] — doc smoke emitted zero-step warning")
        PASS = False

    with redirect_stdout(io.StringIO()) as warn_out:
        warn_args = parse_args([
            "--variant", "small",
            "--iterations", "50",
            "--update-interval", "10",
            "--checkpoint-interval", "25",
            "--batch-size", "256",
            "--aivat-sims", "50",
            "--save-path", warn_path,
            "--device", "cpu",
            "--disable-collapse-canary",
        ])
        warn_result = run_training(warn_args)
    warn_text = warn_out.getvalue()
    print(f"  Default-size short run gradient steps: {warn_result['gradient_steps_taken']}")
    if warn_result["gradient_steps_taken"] == 0:
        print("  [PASS] — default-size short run has 0 regret gradient steps")
    else:
        print("  [FAIL] — default-size short run unexpectedly trained regret")
        PASS = False
    if "[WARN] Training completed with 0 gradient steps" in warn_text:
        print("  [PASS] — zero-step warning emitted")
    else:
        print("  [FAIL] — zero-step warning missing")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False
finally:
    for path in (doc_path, warn_path):
        if path and os.path.exists(path):
            os.unlink(path)

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 5 — Checkpoint save/load round-trip
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 5: Checkpoint save/load round-trip")
print("=" * 60)

try:
    if smoke_path and os.path.exists(smoke_path):
        with redirect_stdout(io.StringIO()):
            fresh_bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                                   aivat_sims=50)
        fresh_opt = torch.optim.Adam(fresh_bot.network.parameters(), lr=1e-4)

        iter_restored = load_checkpoint(smoke_path, fresh_bot, fresh_opt)
        print(f"  Restored iteration: {iter_restored}")

        if iter_restored == 100:
            print("  [PASS] — iteration number matches (100)")
        else:
            print(f"  [FAIL] — expected 100, got {iter_restored}")
            PASS = False

        # Key-by-key tensor equality
        orig_sd = result["bot"].network.state_dict()
        loaded_sd = fresh_bot.network.state_dict()
        all_match = True
        for key in orig_sd:
            if key not in loaded_sd:
                print(f"  [FAIL] — key {key} missing in loaded checkpoint")
                PASS = False
                all_match = False
                break
            if not torch.equal(orig_sd[key].cpu(), loaded_sd[key].cpu()):
                print(f"  [FAIL] — key {key} doesn't match")
                PASS = False
                all_match = False
                break

        if all_match:
            print("  [PASS] — all state_dict keys match")
    else:
        print("  [FAIL] — no checkpoint from Section 4")
        PASS = False

except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 6 — Network state changes after training (anti-stub gate)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 6: Network state changes after training (anti-stub)")
print("=" * 60)

try:
    with redirect_stdout(io.StringIO()):
        stub_bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                              aivat_sims=50)
    stub_bot.network.to(torch.device("cpu"))

    # Capture before state
    before_sd = {k: v.clone() for k, v in stub_bot.network.state_dict().items()}

    stub_opt = torch.optim.Adam(stub_bot.network.parameters(), lr=1e-4)
    sr = ReservoirBuffer(capacity=100_000)
    sv = ReservoirBuffer(capacity=100_000)
    ss = ReservoirBuffer(capacity=100_000)

    # Run 100 traversals + gradient steps
    for i in range(100):
        random.seed(i + 999)
        state = build_initial_state(n_seats=6, hero_seat=i % 6)
        stub_bot.network.eval()
        stub_bot._cfr_recurse(
            state, hero_seat=i % 6, depth=2,
            iteration=i + 1,
            regret_buf=sr, value_buf=sv, sizing_buf=ss,
        )
        if (i + 1) % 20 == 0 and len(sr) >= 32:
            stub_bot.network.train()
            train_step(stub_bot.network, stub_opt, sr, sv, ss, 32, torch.device("cpu"))

    # Capture after state
    after_sd = stub_bot.network.state_dict()

    total_params = len(before_sd)
    changed = 0
    for key in before_sd:
        diff = (after_sd[key].cpu() - before_sd[key].cpu()).abs().max().item()
        if diff > 1e-6:
            changed += 1

    pct = changed / max(total_params, 1) * 100
    print(f"  Parameters changed: {changed}/{total_params} ({pct:.0f}%)")

    if pct >= 50:
        print("  [PASS] — ≥50% of parameters changed")
    else:
        print(f"  [FAIL] — only {pct:.0f}% changed (<50%)")
        PASS = False

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 7 — AIVAT sims configurability
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 7: AIVAT sims is configurable")
print("=" * 60)

try:
    with redirect_stdout(io.StringIO()):
        fast_bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                              aivat_sims=50)
        prod_bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                              aivat_sims=500)
    if fast_bot.aivat_sims == 50 and prod_bot.aivat_sims == 500:
        print("  [PASS] — constructor stores distinct aivat_sims values")
    else:
        print(f"  [FAIL] — aivat_sims mismatch: {fast_bot.aivat_sims}, {prod_bot.aivat_sims}")
        PASS = False
except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 8 — AIVAT leaf recoverable errors are logged
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 8: _aivat_leaf_value recoverable error logging")
print("=" * 60)

try:
    with redirect_stdout(io.StringIO()):
        leaf_bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                              aivat_sims=50)
    bad_state = build_initial_state(n_seats=6, hero_seat=0)
    bad_state.committed_per_seat = []
    bad_state.hole_cards.pop(0, None)
    with redirect_stdout(io.StringIO()) as leaf_out:
        val = leaf_bot._aivat_leaf_value(bad_state, hero_seat=0)
    leaf_text = leaf_out.getvalue()
    print(f"  Returned value: {val}")
    if val == 0.0 and "_aivat_leaf_value caught" in leaf_text:
        print("  [PASS] — recoverable invalid state is logged before returning 0")
    else:
        print("  [FAIL] — invalid state silently returned 0 or did not log")
        PASS = False
except Exception as e:
    print(f"  [FAIL] — unexpected exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 9 — Big blind inference from PlayerView history
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 9: _infer_big_blind")
print("=" * 60)

try:
    base_view = PlayerView(
        me="P0",
        street="preflop",
        position="UTG",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[],
        pot=15,
        to_call=10,
        min_raise=40,
        max_raise=1000,
        legal_actions=[{"type": "fold"}, {"type": "call"}, {"type": "raise", "min": 20, "max": 1000}],
        stacks={"P0": 1000, "P1": 995, "P2": 990},
        opponents=["P1", "P2"],
        history=[{"street": "preflop", "pid": "P3", "type": "call", "amount": 10, "pot_before": 15}],
    )
    inferred = _infer_big_blind(base_view)
    empty_view = PlayerView(
        me=base_view.me,
        street=base_view.street,
        position=base_view.position,
        hole_cards=base_view.hole_cards,
        board=base_view.board,
        pot=base_view.pot,
        to_call=base_view.to_call,
        min_raise=37,
        max_raise=base_view.max_raise,
        legal_actions=base_view.legal_actions,
        stacks=base_view.stacks,
        opponents=base_view.opponents,
        history=[],
    )
    fallback = _infer_big_blind(empty_view)
    print(f"  history inference: {inferred}")
    print(f"  empty-history fallback: {fallback}")
    if abs(inferred - 10) <= 1:
        print("  [PASS] — pot_before=15 infers BB≈10")
    else:
        print("  [FAIL] — did not infer BB from blind pot")
        PASS = False
    if abs(fallback - 10) <= 1:
        print("  [PASS] — empty preflop history falls back to blind pot")
    else:
        print("  [FAIL] — empty history fallback wrong")
        PASS = False
except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 10 — All-in curriculum schedule
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 10: all-in curriculum schedule")
print("=" * 60)

try:
    p_shadow = all_in_policy_probability_for_iteration(19, 20, 80)
    p_start = all_in_policy_probability_for_iteration(20, 20, 80)
    p_mid = all_in_policy_probability_for_iteration(50, 20, 80)
    p_full = all_in_policy_probability_for_iteration(80, 20, 80)
    phases = [
        all_in_phase_for_iteration(19, 20, 80),
        all_in_phase_for_iteration(20, 20, 80),
        all_in_phase_for_iteration(80, 20, 80),
    ]
    print(f"  probs: shadow={p_shadow:.3f}, start={p_start:.3f}, mid={p_mid:.3f}, full={p_full:.3f}")
    print(f"  phases: {phases}")
    if p_shadow == 0.0 and p_start == 0.0 and 0.0 < p_mid < 1.0 and p_full == 1.0:
        print("  [PASS] — staged policy exposure ramps from shadow to full")
    else:
        print("  [FAIL] — staged policy exposure boundaries wrong")
        PASS = False
    if phases == ["shadow", "staged", "full"]:
        print("  [PASS] — phase names match expected boundaries")
    else:
        print("  [FAIL] — phase names wrong")
        PASS = False
except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 11 — Shadow all-in targets without policy exposure
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 11: shadow all-in targets EXCLUDED (Key Change #2)")
print("=" * 60)
# CONTRACT REVERSAL (Key Change #2).  This section previously asserted that in
# the shadow phase all-in was hidden from the policy mask but STILL written as a
# regret target (target flag == 1.0).  Key Change #1's tracer proved that exact
# behavior was the all-in collapse mechanism: shadow-training the `all_in`
# regret row gave it positive regret (action_value[all_in] - EV, with EV taken
# over a mask that excluded all_in), so the row was poisoned long before
# self-play could correct it.  The new contract is the opposite — in the shadow
# phase all-in must be excluded from BOTH the policy mask AND the regret target
# mask whenever any non-all-in action is legal.

try:
    with redirect_stdout(io.StringIO()):
        shadow_bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                                aivat_sims=1)
    shadow_bot._aivat_leaf_value = lambda _state, _hero_seat: 0.0
    seen_masks = []
    original_regret_match = shadow_bot._regret_match

    def spy_regret_match(logits, legal_mask):
        seen_masks.append(list(legal_mask))
        return original_regret_match(logits, legal_mask)

    shadow_bot._regret_match = spy_regret_match
    shadow_state = _DeepCFRGameState(
        pot=15,
        stacks=[1000, 1000],
        committed_per_seat=[0, 10],
        alive=[True, True],
        street="preflop",
        board=[],
        hole_cards={
            0: (("A", "s"), ("K", "s")),
            1: (("2", "c"), ("7", "d")),
        },
        seat_order=[0],
        action_idx=0,
        history_events=[],
        deck_remaining=[],
        big_blind=10,
        ring_order=[0, 1],
    )
    shadow_r = ReservoirBuffer()
    shadow_v = ReservoirBuffer()
    shadow_s = ReservoirBuffer()
    shadow_bot._cfr_recurse(
        shadow_state,
        hero_seat=0,
        depth=1,
        iteration=1,
        regret_buf=shadow_r,
        value_buf=shadow_v,
        sizing_buf=shadow_s,
        allow_all_in=False,
        all_in_policy_probability=0.0,
    )
    all_in_idx = ABSTRACT_ACTIONS.index("all_in")
    root_policy_mask = seen_masks[0]
    target_mask = shadow_r.buffer[0][2]
    # The regret target mask (legal_mask_vec) must agree with the policy mask:
    # every action trained as a target must be the same set the policy expanded.
    target_indices = {i for i in range(NUM_ACTIONS)
                      if float(target_mask[i].item()) == 1.0}
    print(f"  root policy mask: {[ABSTRACT_ACTIONS[i] for i in root_policy_mask]}")
    print(f"  regret target mask: {[ABSTRACT_ACTIONS[i] for i in sorted(target_indices)]}")
    print(f"  all-in target flag: {float(target_mask[all_in_idx].item()):.0f}")
    all_in_excluded = (
        all_in_idx not in root_policy_mask
        and float(target_mask[all_in_idx].item()) == 0.0
    )
    masks_aligned = target_indices == set(root_policy_mask)
    if all_in_excluded and masks_aligned:
        print("  [PASS] — shadow all-in excluded from BOTH policy and regret "
              "target masks; policy and target masks aligned")
    else:
        print("  [FAIL] — shadow all-in contract broken "
              f"(excluded={all_in_excluded}, aligned={masks_aligned})")
        PASS = False
except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 12 — All-in detox only touches output row
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 12: all-in detox isolation")
print("=" * 60)

try:
    with redirect_stdout(io.StringIO()):
        detox_bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                               aivat_sims=1)
    before = {k: v.clone() for k, v in detox_bot.network.state_dict().items()}
    layer_name, all_in_idx = detox_bot.detox_all_in_regret_output()
    after = detox_bot.network.state_dict()

    changed_elsewhere = []
    output_weight = f"{layer_name}.weight"
    output_bias = f"{layer_name}.bias"
    for key, old in before.items():
        diff = after[key] - old
        if key == output_weight:
            mask = torch.ones_like(diff, dtype=torch.bool)
            mask[all_in_idx] = False
            if diff[mask].abs().max().item() > 1e-8:
                changed_elsewhere.append(key)
        elif key == output_bias:
            mask = torch.ones_like(diff, dtype=torch.bool)
            mask[all_in_idx] = False
            if diff[mask].abs().max().item() > 1e-8:
                changed_elsewhere.append(key)
        elif diff.abs().max().item() > 1e-8:
            changed_elsewhere.append(key)

    row_zeroed = after[output_weight][all_in_idx].abs().max().item() == 0.0
    bias_set = abs(float(after[output_bias][all_in_idx].item()) + 2.0) < 1e-8
    print(f"  detox layer: {layer_name}[{all_in_idx}]")
    if row_zeroed and bias_set and not changed_elsewhere:
        print("  [PASS] — detox changed only the all-in regret output row")
    else:
        print(f"  [FAIL] — detox touched unexpected tensors: {changed_elsewhere}")
        PASS = False
except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 13 — Checkpoint promotion preserves safe primary
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 13: checkpoint promotion")
print("=" * 60)

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "deep.pt")
        with redirect_stdout(io.StringIO()):
            promo_bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=False,
                                   aivat_sims=1)
        promo_opt = torch.optim.Adam(promo_bot.network.parameters(), lr=1e-4)
        losses = {"regret": [1.0], "value": [1.0], "sizing": [0.1]}
        save_promoted_checkpoint(ckpt_path, 1, promo_bot, promo_opt, losses, "PASS")
        with torch.no_grad():
            next(promo_bot.network.parameters()).add_(1.0)
        save_promoted_checkpoint(ckpt_path, 2, promo_bot, promo_opt, losses, "WARN")

        primary_iter = torch.load(ckpt_path, map_location="cpu", weights_only=False)["iteration"]
        safe_iter = torch.load(
            safe_checkpoint_path(ckpt_path),
            map_location="cpu",
            weights_only=False,
        )["iteration"]
        warn_iter = torch.load(
            warn_checkpoint_path(ckpt_path, 2),
            map_location="cpu",
            weights_only=False,
        )["iteration"]
    print(f"  primary={primary_iter}, safe={safe_iter}, warn={warn_iter}")
    if primary_iter == 1 and safe_iter == 1 and warn_iter == 2:
        print("  [PASS] — warning checkpoint did not overwrite primary/safe")
    else:
        print("  [FAIL] — checkpoint promotion wrote the wrong files")
        PASS = False
    if (
        classify_canary(0.30, 0.15) == "PASS"
        and classify_canary(0.44, 0.22) == "WARN"
        and classify_canary(0.64, 0.54) == "FAIL"
    ):
        print("  [PASS] — canary thresholds classify pass/warn/fail")
    else:
        print("  [FAIL] — canary threshold classification wrong")
        PASS = False

    # Live-canary health-metric classifier (PFR / avg-raise / strong-all-in).
    # A constructed metric object must classify FAIL over the fail thresholds,
    # WARN in the warn band, and PASS when healthy.  This is the regression that
    # proves PFR — which is intentionally NOT hard-gated in the micro smoke
    # (it saturates at ~85-100% there) — IS enforced in the live canary.
    pfr_fail = classify_extra_canary_metrics({"preflop_pfr": 0.60})[0]
    pfr_warn = classify_extra_canary_metrics({"preflop_pfr": 0.45})[0]
    avg_fail = classify_extra_canary_metrics({"preflop_avg_raise": 30.0})[0]
    avg_warn = classify_extra_canary_metrics({"preflop_avg_raise": 12.0})[0]
    strong_fail = classify_extra_canary_metrics({"strong_all_in": 0.50})[0]
    strong_warn = classify_extra_canary_metrics({"strong_all_in": 0.30})[0]
    healthy = classify_extra_canary_metrics(
        {"preflop_pfr": 0.20, "preflop_avg_raise": 3.0, "strong_all_in": 0.05})[0]
    # And the combined gate: a clean all-in canary + PFR over the fail threshold
    # must still drive the overall checkpoint status to FAIL.
    combined_pfr_fail = _worst_canary_status(
        classify_canary(0.0, 0.0),
        classify_extra_canary_metrics({"preflop_pfr": 0.60})[0],
    )
    print(f"  PFR=60%->{pfr_fail}, PFR=45%->{pfr_warn}, "
          f"avg=30x->{avg_fail}, avg=12x->{avg_warn}, "
          f"strong=50%->{strong_fail}, strong=30%->{strong_warn}, "
          f"healthy->{healthy}, combined(clean all-in + PFR=60%)->{combined_pfr_fail}")
    if (pfr_fail == "FAIL" and pfr_warn == "WARN"
            and avg_fail == "FAIL" and avg_warn == "WARN"
            and strong_fail == "FAIL" and strong_warn == "WARN"
            and healthy == "PASS" and combined_pfr_fail == "FAIL"):
        print("  [PASS] — live-canary metrics classify PFR/avg-raise/strong-all-in "
              "warn/fail; PFR over fail threshold forces overall FAIL")
    else:
        print("  [FAIL] — live-canary metric classification wrong")
        PASS = False

    # Fold-collapse gate: strong_continue is the lone LOW-is-bad extra metric
    # (Key Change #2 aftermath — AA/KK/AKs folded ~100%).  Bands: PASS at
    # >= 80%, WARN in [60%, 80%), FAIL below 60%; the boundaries are pinned
    # exactly.  A missing key must default to HEALTHY (PASS) — a 0.0 default
    # would spuriously FAIL every legacy two-key probe.  The failure reason must
    # NAME the metric (strong_continue) so a fold-collapse abort is attributable
    # separately from an all-in collapse.
    sc_pass = classify_extra_canary_metrics({"strong_continue": 0.80})[0]
    sc_warn_hi = classify_extra_canary_metrics({"strong_continue": 0.799})[0]
    sc_warn_lo = classify_extra_canary_metrics({"strong_continue": 0.60})[0]
    sc_fail = classify_extra_canary_metrics({"strong_continue": 0.599})[0]
    sc_zero_status, sc_zero_reasons, _ = classify_extra_canary_metrics(
        {"strong_continue": 0.0})
    sc_missing = classify_extra_canary_metrics({})[0]
    sc_names = any("strong_continue" in r for r in sc_zero_reasons)
    # Combined gate: a clean all-in canary + a fold-collapsed strong continue
    # must still drive the overall checkpoint status to FAIL.
    combined_fold_fail = _worst_canary_status(
        classify_canary(0.0, 0.0),
        classify_extra_canary_metrics({"strong_continue": 0.0})[0],
    )
    print(f"  strong_continue: 80%->{sc_pass}, 79.9%->{sc_warn_hi}, "
          f"60%->{sc_warn_lo}, 59.9%->{sc_fail}, missing->{sc_missing}, "
          f"reason={sc_zero_reasons}, "
          f"combined(clean all-in + 0%)->{combined_fold_fail}")
    if (sc_pass == "PASS" and sc_warn_hi == "WARN" and sc_warn_lo == "WARN"
            and sc_fail == "FAIL" and sc_zero_status == "FAIL" and sc_names
            and sc_missing == "PASS" and combined_fold_fail == "FAIL"):
        print("  [PASS] — strong_continue classifies PASS>=80/WARN 60-80/FAIL<60, "
              "names the metric, defaults healthy, and forces overall FAIL")
    else:
        print("  [FAIL] — strong_continue fold-collapse classification wrong")
        PASS = False

    # Deferred enforcement: the extra metrics (PFR/avg-raise/strong-all-in/
    # strong-continue) must NOT change the status before iteration >=
    # deploy_iteration, but MUST once the model is mature.  A clean all-in canary
    # + a PFR-over-fail metric:
    #   - early (iter < deploy)  -> PASS (reported only, no false abort)
    #   - mature (iter >= deploy)-> FAIL
    # The always-enforced all-in canary is unaffected by the deploy boundary.
    DEPLOY = 100
    clean_high_pfr = {"raw_all_in": 0.0, "search_all_in": 0.0, "preflop_pfr": 0.60}
    collapsed_allin = {"raw_all_in": 0.99, "search_all_in": 0.99, "preflop_pfr": 0.0}
    early = decide_canary_status(clean_high_pfr, iteration=DEPLOY - 1, deploy_iteration=DEPLOY)
    mature = decide_canary_status(clean_high_pfr, iteration=DEPLOY, deploy_iteration=DEPLOY)
    allin_early = decide_canary_status(collapsed_allin, iteration=DEPLOY - 1, deploy_iteration=DEPLOY)
    print(f"  deferral: PFR=60% early(status,enforced)=({early[0]},{early[5]}), "
          f"mature=({mature[0]},{mature[5]}); all-in collapse early={allin_early[0]}")
    if (early[0] == "PASS" and early[5] is False
            and mature[0] == "FAIL" and mature[5] is True
            and allin_early[0] == "FAIL"):
        print("  [PASS] — extra-metric enforcement deferred to iter >= deploy; "
              "all-in canary still FAILs pre-deploy")
    else:
        print("  [FAIL] — deferred extra-metric enforcement wrong")
        PASS = False

    # The fold-collapse metric follows the SAME deferral, and an all-in collapse
    # must still FAIL independently (proving the two signatures are separable in
    # the live canary, not just the probe): a clean all-in canary + 0%
    # strong_continue is reported-only (PASS) pre-deploy and FAIL once mature.
    fold_metrics = {"raw_all_in": 0.0, "search_all_in": 0.0, "strong_continue": 0.0}
    fold_early = decide_canary_status(fold_metrics, iteration=DEPLOY - 1, deploy_iteration=DEPLOY)
    fold_mature = decide_canary_status(fold_metrics, iteration=DEPLOY, deploy_iteration=DEPLOY)
    print(f"  fold deferral: strong_continue=0% "
          f"early(status,enforced)=({fold_early[0]},{fold_early[5]}), "
          f"mature=({fold_mature[0]},{fold_mature[5]}); "
          f"all-in collapse early(independent)={allin_early[0]}")
    if (fold_early[0] == "PASS" and fold_early[5] is False
            and fold_mature[0] == "FAIL" and fold_mature[5] is True
            and allin_early[0] == "FAIL"):
        print("  [PASS] — strong_continue enforcement deferred to iter >= deploy; "
              "all-in collapse still FAILs independently")
    else:
        print("  [FAIL] — deferred strong_continue enforcement wrong")
        PASS = False
except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 14 — Accelerated all-in curriculum smoke
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 14: accelerated all-in curriculum smoke")
print("=" * 60)

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        curriculum_path = os.path.join(tmpdir, "curriculum.pt")
        original_probe = train_deep_cfr_module.quick_canary_probe

        def passing_probe(*_args, **_kwargs):
            return {"raw_all_in": 0.10, "search_all_in": 0.05}

        train_deep_cfr_module.quick_canary_probe = passing_probe
        try:
            with redirect_stdout(io.StringIO()):
                curriculum_args = parse_args([
                    "--variant", "small",
                    "--iterations", "80",
                    "--update-interval", "20",
                    "--checkpoint-interval", "40",
                    "--batch-size", "32",
                    "--aivat-sims", "1",
                    "--all-in-warmup-iterations", "20",
                    "--all-in-deploy-iteration", "40",
                    "--all-in-full-release-iteration", "80",
                    "--save-path", curriculum_path,
                    "--device", "cpu",
                ])
                curriculum_result = run_training(curriculum_args)
        finally:
            train_deep_cfr_module.quick_canary_probe = original_probe

        primary_exists = os.path.exists(curriculum_path)
        safe_exists = os.path.exists(safe_checkpoint_path(curriculum_path))
        final_iter = curriculum_result["final_iter"]
    print(f"  final_iter={final_iter}, primary={primary_exists}, safe={safe_exists}")
    if final_iter == 80 and primary_exists and safe_exists:
        print("  [PASS] — accelerated curriculum run promoted primary and safe checkpoints")
    else:
        print("  [FAIL] — accelerated curriculum run did not save expected checkpoints")
        PASS = False
except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 15 — Tournament inference masks/caps all-in
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 15: inference all-in guardrails")
print("=" * 60)

try:
    with redirect_stdout(io.StringIO()):
        guard_bot = DeepCFRBot(config=DeepCFRConfig.small(), inference_mode=True,
                               aivat_sims=1)
    guard_bot._weights_loaded = True
    guard_bot.all_in_deploy_iteration = 150
    guard_bot.all_in_full_release_iteration = 350
    guard_bot.training_iteration = 149
    raw_mask = [
        ABSTRACT_ACTIONS.index("fold"),
        ABSTRACT_ACTIONS.index("check_call"),
        ABSTRACT_ACTIONS.index("all_in"),
    ]
    masked = guard_bot._mask_all_in(raw_mask) if guard_bot._inference_all_in_mask_active() else raw_mask

    cap_view = PlayerView(
        me="hero",
        street="preflop",
        position="UTG",
        hole_cards=[("A", "s"), ("K", "s")],
        board=[],
        pot=15,
        to_call=10,
        min_raise=20,
        max_raise=1000,
        legal_actions=[
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": 20, "max": 1000},
        ],
        stacks={"hero": 1000, "opp1": 1000},
        opponents=["opp1"],
        history=[],
    )
    guard_bot.training_iteration = 200
    strategy = [0.0] * NUM_ACTIONS
    strategy[ABSTRACT_ACTIONS.index("check_call")] = 0.2
    strategy[ABSTRACT_ACTIONS.index("all_in")] = 0.8
    capped = guard_bot._cap_all_in_probability(strategy, raw_mask, cap_view)
    all_in_prob = capped[ABSTRACT_ACTIONS.index("all_in")]
    print(f"  masked labels: {[ABSTRACT_ACTIONS[i] for i in masked]}")
    print(f"  staged all-in prob after cap: {all_in_prob:.3f}")
    if ABSTRACT_ACTIONS.index("all_in") not in masked and 0.0 < all_in_prob <= 0.15:
        print("  [PASS] — inference masks before deploy and caps during staged release")
    else:
        print("  [FAIL] — inference guardrails failed")
        PASS = False
except Exception as e:
    print(f"  [FAIL] — exception: {type(e).__name__}: {e}")
    PASS = False

print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 16 — Existing sanity scripts still pass
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Section 16: sanity_deep_cfr.py --variant small regression check")
print("=" * 60)

try:
    result_proc = subprocess.run(
        [sys.executable, "sanity_deep_cfr.py", "--variant", "small"],
        capture_output=True, text=True, timeout=300,
    )
    if result_proc.returncode == 0:
        print("  [PASS] — sanity_deep_cfr.py --variant small passed")
    else:
        print("  [FAIL] — sanity_deep_cfr.py regressed")
        print("  STDOUT (last 10 lines):")
        for line in result_proc.stdout.strip().split("\n")[-10:]:
            print(f"    {line}")
        if result_proc.stderr:
            print("  STDERR (last 5 lines):")
            for line in result_proc.stderr.strip().split("\n")[-5:]:
                print(f"    {line}")
        PASS = False

except subprocess.TimeoutExpired:
    print("  [FAIL] — sanity_deep_cfr.py timed out (>300s)")
    PASS = False
except Exception as e:
    print(f"  [FAIL] — could not run: {e}")
    PASS = False

print()

# ═══════════════════════════════════════════════════════════════════════════════
#  Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

if smoke_path and os.path.exists(smoke_path):
    os.unlink(smoke_path)

# ═══════════════════════════════════════════════════════════════════════════════
#  OVERALL
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
if PASS:
    print("ALL CHECKS PASSED [PASS]")
else:
    print("SOME CHECKS FAILED [FAIL]")
    sys.exit(1)
print("=" * 60)
