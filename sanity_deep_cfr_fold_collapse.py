#!/usr/bin/env python3
"""
sanity_deep_cfr_fold_collapse.py — fold-collapse health-gate regression.

After Key Change #2 a 25k diagnostic checkpoint stopped all-in collapsing but
fold-collapsed instead: AA/KK/AKs were folded ~100%, yet the original
--fail-on-unhealthy gate reported HEALTHY because it only watched the all-in
signature (all-in / PFR / avg-raise / strong-all-in).  This file pins the
fold-collapse gate that closes that hole.

It is intentionally TRAINING-FREE — every check runs against constructed action
distributions and a fold/call/all-in stub bot, so it belongs in the fast
default validation tier (no real run, no model/checkpoint files touched).

  CHECK 1  Negative (probe gate): a 100%-fold strong-hand distribution trips the
           probe HEALTH GATE, AND the tripping row is specifically strong-hand
           continue — the all-in rows stay OK (fold collapse, not all-in
           collapse).
  CHECK 2  Positive (probe gate): a healthy strong-hand distribution (continues
           AA/KK/AKs) passes the strong-continue gate.
  CHECK 3  Separability: an all-in-collapse distribution trips the all-in row
           but NOT strong-continue (100% continue) — the two collapse
           signatures are detected independently.
  CHECK 4  End-to-end exit code: probe_deep_cfr.run_probe(--fail-on-unhealthy)
           exits nonzero on a fold-only stub bot and zero on a (healthy)
           call-only stub bot — exercising the real CLI gate and exit path.
  CHECK 5  Live-training canary: classify_extra_canary_metrics flags a 0%
           strong-continue as FAIL and names ``strong_continue`` in the reason,
           while a healthy/missing value passes.
  CHECK 6  Live-training canary log: format_canary_metrics (the single source
           for the [CANARY]/[WARN]/abort lines) renders the strong-continue %
           — 0.0% on a fold collapse, the healthy 100.0% default on a legacy
           two-key probe.  Verifies the log rendering without a training run.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

from core.bot_api import Action

import probe_deep_cfr
from probe_deep_cfr import (
    _evaluate_health, _print_health_gate, _new_stats, _record, _view,
    _strong_continue_pct, run_probe,
)
from training.train_deep_cfr import (
    classify_extra_canary_metrics, format_canary_metrics,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stub_action(kind, view):
    """Build the action a fixed-policy stub bot returns for ``view``."""
    if kind == "all_in":
        # A max-stack raise — probe._is_all_in() treats amount >= max_raise as
        # an effective all-in, matching the bucket->max collapse signature.
        return Action(type="raise", amount=view.max_raise)
    return Action(type=kind, amount=None)


def _stats_from_stub(kind, n=50):
    """Collect a _new_stats distribution from a fixed-policy stub over n spots."""
    stats = _new_stats()
    view = _view([("A", "h"), ("A", "s")], position="BTN",
                 pot=15, to_call=10, stack=500)
    for _ in range(n):
        _record(stats, _stub_action(kind, view), view)
    return stats


def _row(rows, needle):
    """Return the (label, value, thr, direction, tripped) row matching needle."""
    for row in rows:
        if needle in row[0]:
            return row
    raise AssertionError(f"no health-gate row matching {needle!r}")


class _StubBot:
    """Minimal fixed-policy bot for the end-to-end probe path (no network)."""

    def __init__(self, kind):
        self._kind = kind
        self.search_depth = 1  # _load_bot may zero this under --disable-search

    def act(self, view):
        return _stub_action(self._kind, view)


def _probe_args(**overrides):
    base = dict(weights="<stub>", samples=20, chips=500, seed=42,
                disable_search=True, trace=0, fail_on_unhealthy=True)
    base.update(overrides)
    return argparse.Namespace(**base)


def _run_probe_with_stub(kind):
    """Run the real run_probe() with _load_bot patched to a fixed-policy stub.

    Returns the exit code.  Patches the module attribute so run_probe's
    module-global _load_bot reference resolves to the stub; always restored.
    """
    original = probe_deep_cfr._load_bot
    probe_deep_cfr._load_bot = lambda _w, _ds: _StubBot(kind)
    try:
        with redirect_stdout(io.StringIO()):
            return run_probe(_probe_args())
    finally:
        probe_deep_cfr._load_bot = original


# ── CHECK 1: fold collapse trips strong-continue, not the all-in rows ─────────

def check_negative_probe_gate(pass_state):
    pre = _stats_from_stub("call")        # healthy preflop (no shoves/raises)
    strong = _stats_from_stub("fold")     # 100% fold → 0% continue
    rows = _evaluate_health(pre, strong)
    cont = _row(rows, "strong-hand continue")
    allin = _row(rows, "preflop all-in")
    strong_allin = _row(rows, "strong-hand all-in")

    cont_tripped = cont[4] is True
    allin_clean = allin[4] is False and strong_allin[4] is False
    with redirect_stdout(io.StringIO()):
        healthy = _print_health_gate(pre, strong)
    ok = cont_tripped and allin_clean and healthy is False
    pass_state[0] &= ok
    print(f"[CHECK 1] {'PASS' if ok else 'FAIL'} — fold collapse: "
          f"strong_continue={_strong_continue_pct(strong):.0f}% tripped="
          f"{cont_tripped}; all-in rows clean={allin_clean}; "
          f"gate healthy={healthy} (expect False)")


# ── CHECK 2: healthy strong-hand continue passes the gate ─────────────────────

def check_positive_probe_gate(pass_state):
    pre = _stats_from_stub("call")
    strong = _stats_from_stub("call")     # 100% continue
    rows = _evaluate_health(pre, strong)
    cont = _row(rows, "strong-hand continue")
    with redirect_stdout(io.StringIO()):
        healthy = _print_health_gate(pre, strong)
    ok = cont[4] is False and healthy is True
    pass_state[0] &= ok
    print(f"[CHECK 2] {'PASS' if ok else 'FAIL'} — healthy strong-continue: "
          f"strong_continue={_strong_continue_pct(strong):.0f}% tripped={cont[4]} "
          f"(expect False); gate healthy={healthy} (expect True)")


# ── CHECK 3: all-in collapse is detected separately from fold collapse ────────

def check_allin_independent(pass_state):
    pre = _stats_from_stub("all_in")      # shoves everything
    strong = _stats_from_stub("all_in")   # shoves AA/KK/AKs (still continues)
    rows = _evaluate_health(pre, strong)
    pre_allin = _row(rows, "preflop all-in")
    strong_allin = _row(rows, "strong-hand all-in")
    cont = _row(rows, "strong-hand continue")
    allin_tripped = pre_allin[4] is True and strong_allin[4] is True
    cont_clean = cont[4] is False  # shoving is still continuing
    ok = allin_tripped and cont_clean
    pass_state[0] &= ok
    print(f"[CHECK 3] {'PASS' if ok else 'FAIL'} — all-in collapse trips all-in "
          f"rows={allin_tripped} while strong_continue stays clean={cont_clean} "
          f"(continue={_strong_continue_pct(strong):.0f}%); signatures separable")


# ── CHECK 4: real run_probe(--fail-on-unhealthy) exit codes ───────────────────

def check_end_to_end_exit_code(pass_state):
    fold_rc = _run_probe_with_stub("fold")   # fold collapse → nonzero
    call_rc = _run_probe_with_stub("call")   # healthy        → zero
    ok = fold_rc == 1 and call_rc == 0
    pass_state[0] &= ok
    print(f"[CHECK 4] {'PASS' if ok else 'FAIL'} — run_probe(--fail-on-unhealthy): "
          f"fold-only stub rc={fold_rc} (expect 1), call-only stub rc={call_rc} "
          f"(expect 0)")


# ── CHECK 5: live-training canary classifier ──────────────────────────────────

def check_live_canary_classifier(pass_state):
    fail_status, fail_reasons, _ = classify_extra_canary_metrics(
        {"strong_continue": 0.0})
    warn_status = classify_extra_canary_metrics({"strong_continue": 0.70})[0]
    pass_status = classify_extra_canary_metrics({"strong_continue": 0.95})[0]
    missing_status = classify_extra_canary_metrics(
        {"raw_all_in": 0.99, "search_all_in": 0.99})[0]  # legacy probe → healthy
    names_metric = any("strong_continue" in r for r in fail_reasons)
    ok = (fail_status == "FAIL" and warn_status == "WARN"
          and pass_status == "PASS" and missing_status == "PASS"
          and names_metric)
    pass_state[0] &= ok
    print(f"[CHECK 5] {'PASS' if ok else 'FAIL'} — live canary: 0%->{fail_status}, "
          f"70%->{warn_status}, 95%->{pass_status}, legacy/missing->{missing_status}; "
          f"reason names metric={names_metric} ({fail_reasons})")


# ── CHECK 6: live-training canary log renders the strong-continue % ──────────

def check_live_canary_log_renders(pass_state):
    # The live [CANARY]/[WARN]/abort log line must surface the strong-hand
    # continue percentage.  format_canary_metrics is the single source feeding
    # all three branches, so testing it here proves the rendering without a
    # training run.  A fold-collapsed probe (0% continue) must render "0.0%";
    # a legacy two-key probe must render the healthy default (100.0%).
    fold_line = format_canary_metrics(
        {"raw_all_in": 0.0, "search_all_in": 0.0, "strong_continue": 0.0})
    legacy_line = format_canary_metrics(
        {"raw_all_in": 0.1, "search_all_in": 0.05})
    fold_ok = "strong_continue=0.0%" in fold_line
    legacy_ok = "strong_continue=100.0%" in legacy_line
    ok = fold_ok and legacy_ok
    pass_state[0] &= ok
    print(f"[CHECK 6] {'PASS' if ok else 'FAIL'} — live canary log renders "
          f"strong_continue: fold-collapsed line has 0.0% ({fold_ok}), legacy "
          f"probe defaults to 100.0% ({legacy_ok})")
    print(f"           fold line: {fold_line}")


def run():
    pass_state = [True]
    check_negative_probe_gate(pass_state)
    check_positive_probe_gate(pass_state)
    check_allin_independent(pass_state)
    check_end_to_end_exit_code(pass_state)
    check_live_canary_classifier(pass_state)
    check_live_canary_log_renders(pass_state)
    print("=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED [PASS]' if pass_state[0] else 'SOME CHECKS FAILED [FAIL]'}")
    return pass_state[0]


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
