#!/usr/bin/env python3
"""
sanity_validation_ladder.py — canonical pre-training validation gate.

Run this BEFORE a clean CFR / Deep CFR retrain.  It executes a tiered ladder of
the repo's standalone ``sanity_*.py`` scripts — engine truth → abstraction →
feature schema → chip accounting → smoke training → eval readiness — and exits
nonzero if any *selected* gate fails.  The point is to confirm engine
semantics, abstraction, feature schemas, chip accounting, eval fairness, and
(optionally) smoke-training gates all pass before spending hours on a real run.

GATE CONTRACT
-------------
A gate is FAILED iff its subprocess returns a nonzero exit code OR its combined
stdout/stderr contains the exact marker ``SOME CHECKS FAILED``.  Most sanity
scripts use ``sys.exit(0 if run() else 1)``; a few (e.g. sanity_test_hand,
sanity_test_followon) print the ``OVERALL: ... SOME CHECKS FAILED`` line without
calling ``sys.exit`` — both signals are handled.  Diagnostic-only scripts (e.g.
sanity_action_order) have no failure concept and pass as long as they run
cleanly (rc 0, no marker).

To add a new gate: append a ``Gate(...)`` to ``LADDER`` and make the script
honor either failure signal above.  Keep the list in deterministic tier order.

USAGE
-----
  .venv/bin/python sanity_validation_ladder.py --path {cfr,deep-cfr,both}
  .venv/bin/python sanity_validation_ladder.py --path both --full
  .venv/bin/python sanity_validation_ladder.py --path both --keep-going

  Default mode runs the fast/medium gates and SKIPS the slow smoke-training and
  full-eval gates (Tier 5 + sanity_eval).  ``--full`` adds them.  ``--keep-going``
  runs every selected gate even after a failure instead of halting at the first.

  NOTE: ``--full --path both`` is the expensive mode (~11 min on an M5 Max with
  the smoke-sized budgets; longer under load): sanity_eval.py re-runs the training
  gates internally, so sanity_train_cfr / sanity_train_deep_cfr each run ~twice.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

FAILURE_MARKER = "SOME CHECKS FAILED"


# ─────────────────────────────────────────────────────────────────────────────
#  Gate registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Gate:
    name: str
    tier: int
    applies: str               # "all" | "cfr" | "deep-cfr"
    kind: str = "script"       # "script" | "pycompile" | "imports"
    script: str | None = None  # sanity_*.py filename (kind == "script")
    slow: bool = False         # skipped unless --full
    timeout: int = 600         # seconds
    args: list[str] = field(default_factory=list)  # extra argv for the script


TIER_TITLES = {
    0: "static / import health",
    1: "engine truth",
    2: "CFR / Deep CFR reconstruction & abstraction",
    3: "feature / schema consistency",
    4: "chip / value accounting",
    5: "smoke training gates (slow)",
    6: "eval / readiness",
}

# Deterministic tier order; registry order within a tier is the run order.
LADDER: list[Gate] = [
    # Tier 0 — static / import health (always; fast)
    Gate("py_compile", 0, "all", kind="pycompile", timeout=300),
    Gate("import_smoke", 0, "all", kind="imports", timeout=180),

    # Tier 1 — engine truth (always; fast)
    Gate("sanity_engine_allin_closure", 1, "all",
         script="sanity_engine_allin_closure.py", timeout=300),
    Gate("sanity_action_order", 1, "all",
         script="sanity_action_order.py", timeout=120),
    Gate("sanity_test_hand", 1, "all",
         script="sanity_test_hand.py", timeout=180),
    Gate("sanity_test_followon", 1, "all",
         script="sanity_test_followon.py", timeout=300),
    Gate("sanity_smart_bot_river", 1, "all",
         script="sanity_smart_bot_river.py", timeout=120),

    # Tier 2 — CFR / Deep CFR reconstruction & abstraction.
    #   Covers to_call / pot+commit reconstruction, legal-action parity, and
    #   CFR abstract→concrete bet sizing.  Deep CFR abstract→concrete sizing
    #   parity is covered by sanity_deep_cfr_fixes (Tier 3).
    Gate("sanity_review_findings", 2, "all",
         script="sanity_review_findings.py", timeout=600),
    Gate("sanity_cfr_sizing_parity", 2, "cfr",
         script="sanity_cfr_sizing_parity.py", timeout=300),
    Gate("sanity_cfr_equity", 2, "cfr",
         script="sanity_cfr_equity.py", timeout=600),

    # Tier 3 — feature / schema consistency.
    #   Deep CFR train-vs-inference feature semantics + opponent
    #   committed / can-act / all-in input slots (FIX_REPORT #3/#4); the
    #   decision tracer (Key Change #1); and the Key Change #2 regret-mask
    #   plumbing (fast mask checks only — the micro-training smoke is the
    #   slow Tier 5 gate sanity_deep_cfr_allin_micro).
    Gate("sanity_deep_cfr_fixes", 3, "deep-cfr",
         script="sanity_deep_cfr_fixes.py", timeout=600),
    Gate("sanity_deep_cfr_trace", 3, "deep-cfr",
         script="sanity_deep_cfr_trace.py", timeout=120),
    Gate("sanity_deep_cfr_allin_masks", 3, "deep-cfr",
         script="sanity_deep_cfr_allin_curriculum.py", args=["--mode", "fast"],
         timeout=300),
    # Fold-collapse health gate (post-Key-Change-#2): strong-hand continue
    # gate in the probe + live training canary.  Training-free, so it runs in
    # the default fast tier (not gated behind --full).
    Gate("sanity_deep_cfr_fold_collapse", 3, "deep-cfr",
         script="sanity_deep_cfr_fold_collapse.py", timeout=180),

    # Tier 4 — chip / value accounting (always)
    Gate("sanity_aivat", 4, "all",
         script="sanity_aivat.py", timeout=600),
    Gate("sanity_icm_payouts", 4, "all",
         script="sanity_icm_payouts.py", timeout=300),
    Gate("sanity_preflop_strength", 4, "all",
         script="sanity_preflop_strength.py", timeout=300),

    # Tier 5 — smoke training gates (SLOW; --full only)
    Gate("sanity_deep_cfr_allin_micro", 5, "deep-cfr", slow=True,
         script="sanity_deep_cfr_allin_curriculum.py", args=["--mode", "micro"],
         timeout=900),
    Gate("sanity_deep_cfr", 5, "deep-cfr", slow=True,
         script="sanity_deep_cfr.py", timeout=1800),
    Gate("sanity_train_deep_cfr", 5, "deep-cfr", slow=True,
         script="sanity_train_deep_cfr.py", timeout=1800),
    Gate("sanity_train_deep_cfr_abort", 5, "deep-cfr", slow=True,
         script="sanity_train_deep_cfr_abort.py", timeout=900),
    Gate("sanity_train_cfr", 5, "cfr", slow=True,
         script="sanity_train_cfr.py", timeout=1800),

    # Tier 6 — eval / readiness
    Gate("sanity_run_eval_fixes", 6, "all",
         script="sanity_run_eval_fixes.py", timeout=600),
    Gate("sanity_eval", 6, "all", slow=True,
         script="sanity_eval.py", timeout=3600),
]


IMPORT_SMOKE_CODE = (
    "from core.engine import Table; "
    "from bots import create_bot; "
    "from bots.cfr_bot import CFRBot; "
    "from bots.deep_cfr_bot import DeepCFRBot; "
    "print('import smoke OK')"
)


# ─────────────────────────────────────────────────────────────────────────────
#  Pass / fail classifier (pure function + self-test)
# ─────────────────────────────────────────────────────────────────────────────

def classify(returncode: int, stdout: str, stderr: str) -> str:
    """Return "PASS" or "FAIL" for a finished gate subprocess.

    A gate fails on a nonzero exit code OR if it printed the exact OVERALL
    failure marker.  We anchor on the full ``SOME CHECKS FAILED`` string and
    never on a bare ``FAIL`` — scripts print ``[FAIL]`` check labels and
    "must not FAIL"-style descriptions during *passing* runs.
    """
    if returncode != 0:
        return "FAIL"
    if FAILURE_MARKER in (stdout or "") or FAILURE_MARKER in (stderr or ""):
        return "FAIL"
    return "PASS"


def _selftest_classify() -> None:
    """Guard the one piece whose silent breakage gives false greens."""
    cases = [
        ((0, "OVERALL: ALL CHECKS PASSED [PASS]", ""), "PASS"),
        ((1, "", ""), "FAIL"),
        ((0, "OVERALL: SOME CHECKS FAILED [FAIL]", ""), "FAIL"),
        # Passing run whose text contains '[FAIL]' labels / 'must not FAIL':
        ((0, "PREMIUM — must not FAIL\n  [FAIL] only-a-label\nALL CHECKS PASSED", ""),
         "PASS"),
    ]
    for (rc, out, err), expected in cases:
        got = classify(rc, out, err)
        if got != expected:
            print(f"[ladder] FATAL: classify self-test failed for "
                  f"(rc={rc}, out={out!r}) → {got}, expected {expected}")
            sys.exit(2)


# ─────────────────────────────────────────────────────────────────────────────
#  Gate selection & command building
# ─────────────────────────────────────────────────────────────────────────────

def gate_selected_by_path(gate: Gate, path: str) -> bool:
    if gate.applies == "all":
        return True
    if gate.applies == "cfr":
        return path in ("cfr", "both")
    if gate.applies == "deep-cfr":
        return path in ("deep-cfr", "both")
    return False


def _pycompile_targets() -> list[str]:
    files: list[str] = []
    for pattern in ("core/*.py", "bots/*.py", "training/*.py", "run_*.py"):
        files.extend(glob.glob(os.path.join(REPO_ROOT, pattern)))
    return sorted(set(files))


def build_command(gate: Gate) -> tuple[list[str] | None, str | None]:
    """Return (argv, missing_path).  Exactly one is non-None."""
    if gate.kind == "pycompile":
        return [sys.executable, "-m", "py_compile", *_pycompile_targets()], None
    if gate.kind == "imports":
        return [sys.executable, "-c", IMPORT_SMOKE_CODE], None
    # kind == "script"
    script_path = os.path.join(REPO_ROOT, gate.script)
    if not os.path.exists(script_path):
        return None, gate.script
    return [sys.executable, script_path, *gate.args], None


# ─────────────────────────────────────────────────────────────────────────────
#  Result record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    gate: Gate
    status: str            # "PASS" | "FAIL" | "SKIP"
    elapsed: float = 0.0
    reason: str = ""       # skip reason / failure note


def _indent(text: str, prefix: str = "      | ") -> str:
    return "\n".join(prefix + line for line in (text or "").rstrip().splitlines())


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ladder(path: str, full: bool, keep_going: bool) -> int:
    _selftest_classify()

    print("=" * 70)
    print(f"PRE-TRAINING VALIDATION LADDER   path={path}  full={full}  "
          f"keep-going={keep_going}")
    print("=" * 70)
    if full:
        print("WARNING: --full runs the slow smoke-training + full-eval gates.")
        print("  For --path both expect ~11 min on an M5 Max (longer under load) —")
        print("  sanity_eval.py re-runs the training gates internally "
              "(sanity_train_cfr / sanity_train_deep_cfr run ~twice).")
    else:
        print("Default mode: fast/medium gates only. Slow gates (Tier 5 + "
              "sanity_eval) are SKIPPED — re-run with --full to include them.")
    print()

    results: list[Result] = []
    halted = False

    for gate in LADDER:
        # 1. Halted (no --keep-going, an earlier gate failed).
        if halted:
            results.append(Result(gate, "SKIP", 0.0,
                                   "not run (halted on earlier failure; "
                                   "use --keep-going to run all)"))
            continue

        # 2. Not selected by --path.
        if not gate_selected_by_path(gate, path):
            results.append(Result(gate, "SKIP", 0.0,
                                   f"{gate.applies}-path gate; not selected by "
                                   f"--path {path}"))
            continue

        # 3. Slow gate without --full.
        if gate.slow and not full:
            results.append(Result(gate, "SKIP", 0.0,
                                   "slow gate; enable with --full"))
            continue

        # 4. Build command (missing script → loud FAIL, never a silent skip).
        cmd, missing = build_command(gate)
        if missing is not None:
            print(f"[FAIL] {gate.name}: registry script not found: {missing}")
            print()
            results.append(Result(gate, "FAIL", 0.0,
                                   f"registry script not found: {missing}"))
            if not keep_going:
                halted = True
            continue

        # 5. Run it (liveness print first; output is captured).
        print(f"running  {gate.name}  (tier {gate.tier}, timeout {gate.timeout}s) ...",
              flush=True)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=gate.timeout, cwd=REPO_ROOT)
            elapsed = time.monotonic() - t0
            status = classify(proc.returncode, proc.stdout, proc.stderr)
            if status == "FAIL":
                print(f"[FAIL] {gate.name}  ({elapsed:.1f}s, rc={proc.returncode}) "
                      f"— captured output:")
                if proc.stdout.strip():
                    print(_indent(proc.stdout))
                if proc.stderr.strip():
                    print("      |--- stderr ---")
                    print(_indent(proc.stderr))
                results.append(Result(gate, "FAIL", elapsed,
                                       f"rc={proc.returncode}"))
            else:
                print(f"[PASS] {gate.name}  ({elapsed:.1f}s)")
                results.append(Result(gate, "PASS", elapsed))
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            print(f"[FAIL] {gate.name}  — TIMEOUT after {gate.timeout}s")
            partial = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            if partial.strip():
                print("      |--- partial stdout ---")
                print(_indent(partial))
            results.append(Result(gate, "FAIL", elapsed,
                                   f"timeout after {gate.timeout}s"))

        print()
        if results[-1].status == "FAIL" and not keep_going:
            halted = True

    return _summarize(results, path, full, keep_going)


def _summarize(results: list[Result], path: str, full: bool,
               keep_going: bool) -> int:
    print("=" * 70)
    print(f"VALIDATION LADDER SUMMARY   path={path}  full={full}  "
          f"keep-going={keep_going}")
    print("=" * 70)

    by_tier: dict[int, list[Result]] = {}
    for res in results:
        by_tier.setdefault(res.gate.tier, []).append(res)

    badge = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}
    for tier in sorted(by_tier):
        print(f"\nTier {tier} — {TIER_TITLES.get(tier, '')}")
        for res in by_tier[tier]:
            line = f"  {badge[res.status]} {res.gate.name:<32}"
            if res.status == "SKIP":
                line += f"  {res.reason}"
            else:
                line += f"  {res.elapsed:6.1f}s"
                if res.status == "FAIL" and res.reason:
                    line += f"  ({res.reason})"
            print(line)

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    total_elapsed = sum(r.elapsed for r in results)

    print("\n" + "-" * 70)
    overall = "PASS" if failed == 0 else "FAIL"
    print(f"RESULT: {overall}   "
          f"({passed} passed, {failed} failed, {skipped} skipped)   "
          f"total {total_elapsed:.1f}s")
    if failed:
        names = ", ".join(r.gate.name for r in results if r.status == "FAIL")
        print(f"FAILED GATES: {names}")
    print("=" * 70)

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paranoid pre-training validation ladder. Run before a "
                    "clean CFR / Deep CFR retrain.")
    parser.add_argument("--path", choices=["cfr", "deep-cfr", "both"],
                        default="both",
                        help="Which retrain path to validate (default: both).")
    parser.add_argument("--full", action="store_true",
                        help="Include slow smoke-training + full-eval gates "
                             "(Tier 5 and sanity_eval).")
    parser.add_argument("--keep-going", action="store_true",
                        help="Run every selected gate even after a failure "
                             "instead of halting at the first.")
    args = parser.parse_args()
    return run_ladder(args.path, args.full, args.keep_going)


if __name__ == "__main__":
    sys.exit(main())
