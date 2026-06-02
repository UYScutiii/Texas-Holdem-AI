#!/usr/bin/env python3
"""
probe_deep_cfr.py — action-distribution probe for a Deep CFR checkpoint.

This is diagnostic only. It does not train and does not modify checkpoints.
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from collections import Counter
from contextlib import redirect_stdout

from core.bot_api import PlayerView
from core.engine import _FULL_DECK
from bots.deep_cfr_bot import DeepCFRBot, ABSTRACT_ACTIONS


POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
RANKS = "23456789TJQKA"


def _load_bot(weights_path: str, disable_search: bool) -> DeepCFRBot:
    with redirect_stdout(io.StringIO()):
        bot = DeepCFRBot(weights_path=weights_path, inference_mode=True)
    if disable_search:
        bot.search_depth = 0
    return bot


def _deck_without(cards):
    used = set(cards)
    deck = [c for c in _FULL_DECK if c not in used]
    random.shuffle(deck)
    return deck


def _random_hole():
    deck = list(_FULL_DECK)
    random.shuffle(deck)
    return [deck.pop(), deck.pop()]


def _random_board(exclude, n=3):
    deck = _deck_without(exclude)
    return [deck.pop() for _ in range(n)]


def _view(hole, *, board=None, street="preflop", position="BTN",
          pot=15, to_call=10, stack=500, n_opponents=5):
    board = list(board or [])
    opponents = [f"opp{i}" for i in range(1, n_opponents + 1)]
    stacks = {"hero": stack}
    for op in opponents:
        stacks[op] = stack

    if to_call > 0:
        legal = [
            {"type": "fold"},
            {"type": "call"},
            {"type": "raise", "min": max(to_call + 10, 20), "max": stack},
        ]
    else:
        legal = [
            {"type": "check"},
            {"type": "bet", "min": 10, "max": stack},
        ]

    return PlayerView(
        me="hero",
        street=street,
        position=position,
        hole_cards=list(hole),
        board=board,
        pot=pot,
        to_call=to_call,
        min_raise=legal[-1].get("min", 0),
        max_raise=legal[-1].get("max", 0),
        legal_actions=legal,
        stacks=stacks,
        opponents=opponents,
        history=[],
    )


def _is_vpip(action) -> bool:
    return action.type in ("call", "bet", "raise", "all_in")


def _is_raise(action) -> bool:
    return action.type in ("bet", "raise", "all_in")


def _is_all_in(action, view) -> bool:
    if action.type == "all_in":
        return True
    if action.type not in ("bet", "raise"):
        return False
    return action.amount is not None and action.amount >= view.max_raise


def _record(stats, action, view):
    stats["total"] += 1
    stats["actions"][action.type] += 1
    if _is_vpip(action):
        stats["vpip"] += 1
    if _is_raise(action):
        stats["pfr"] += 1
    if _is_all_in(action, view):
        stats["all_in"] += 1
    if action.type == "fold":
        stats["fold"] += 1
    if action.type in ("bet", "raise") and action.amount is not None:
        stats["sizes"].append(action.amount / max(view.pot, 1))


def _new_stats():
    return {
        "total": 0,
        "actions": Counter(),
        "vpip": 0,
        "pfr": 0,
        "all_in": 0,
        "fold": 0,
        "sizes": [],
    }


def _pct(num, den):
    return num / den * 100 if den else 0.0


# ── Decision tracing (diagnostic, --trace N) ────────────────────────────────

def _fmt_cards(cards):
    out = []
    for c in cards:
        try:
            r, s = c
            out.append(f"{r}{s}")
        except (TypeError, ValueError):
            out.append(str(c))
    return "".join(out)


def _fmt_probs(d):
    if not d:
        return "(none)"
    return "  ".join(f"{ABSTRACT_ACTIONS[i]}={d[i]:.3f}" for i in sorted(d))


def _fmt_logits(d):
    if not d:
        return "(none)"
    return "  ".join(f"{ABSTRACT_ACTIONS[i]}={d[i]:+.2f}" for i in sorted(d))


def _print_trace(section, n, view, trace):
    """Compact per-decision trace. Only legal abstract actions are dumped."""
    print(f"  [{section} #{n}] hole={_fmt_cards(view.hole_cards)} "
          f"pos={view.position} street={view.street} pot={view.pot} "
          f"to_call={view.to_call} bb={trace.big_blind}")
    masked = "   (all_in hidden by inference mask)" if trace.all_in_masked else ""
    print(f"    legal abstract : {trace.legal_abstract_labels}{masked}")
    print(f"    regret logits  : {_fmt_logits(trace.regret_logits)}")
    print(f"    P(pre-search)  : {_fmt_probs(trace.probs_before_search)}")
    if trace.probs_after_search is not None:
        print(f"    P(post-search) : {_fmt_probs(trace.probs_after_search)}")
    prev = trace.probs_after_search or trace.probs_before_search
    if trace.probs_after_cap != prev:
        print(f"    P(post cap)    : {_fmt_probs(trace.probs_after_cap)}")
    print(f"    sizing head    : {trace.sizing_value:.3f}")
    print(f"    SELECTED       : [{trace.selected_abstract_index}] "
          f"{trace.selected_abstract_label}")
    amt = trace.final_action_amount
    amt_str = f" {amt}" if amt is not None else ""
    print(f"    final action   : {trace.final_action_type}{amt_str}")
    if trace.final_is_all_in:
        if trace.selected_is_all_in:
            print("    >> EFFECTIVE ALL-IN: came from the abstract 'all_in' row")
        else:
            print(f"    >> EFFECTIVE ALL-IN: bucket '{trace.selected_abstract_label}'"
                  f" converted to a max-stack raise (NOT the 'all_in' row)")
    print()


def _print_stats(title, stats, *, healthy=None):
    n = stats["total"]
    avg_size = sum(stats["sizes"]) / len(stats["sizes"]) if stats["sizes"] else 0.0
    print(title)
    print("-" * len(title))
    print(f"Samples:          {n}")
    print(f"Actions:          {dict(stats['actions'])}")
    print(f"VPIP:             {_pct(stats['vpip'], n):5.1f}%")
    print(f"PFR/bet freq:     {_pct(stats['pfr'], n):5.1f}%")
    print(f"All-in freq:      {_pct(stats['all_in'], n):5.1f}%")
    print(f"Fold freq:        {_pct(stats['fold'], n):5.1f}%")
    print(f"Avg bet/raise:    {avg_size:5.2f}x pot")
    if healthy:
        for label, ok, detail in healthy(stats):
            mark = "OK" if ok else "FLAG"
            print(f"{mark}: {label} — {detail}")
    print()


def _preflop_health(stats):
    n = stats["total"]
    vpip = _pct(stats["vpip"], n)
    pfr = _pct(stats["pfr"], n)
    all_in = _pct(stats["all_in"], n)
    fold = _pct(stats["fold"], n)
    return [
        ("VPIP", 20 <= vpip <= 40, "healthy rough target 20-40%"),
        ("PFR", 15 <= pfr <= 25, "healthy rough target 15-25%"),
        ("all-in", all_in < 5, "healthy rough target <5%"),
        ("fold", 60 <= fold <= 80, "healthy rough target 60-80%"),
    ]


# ── Failure-mode gate (--fail-on-unhealthy) ─────────────────────────────────
#
# Hard thresholds that flag the two known collapse signatures.  Most metrics are
# "high is bad" — a metric trips when value >= threshold (the all-in-collapse
# signature).  ``strong_continue_min`` is the lone "low is bad" gate: it trips
# when the strong-hand continue rate falls BELOW the threshold (the fold-collapse
# signature, Key Change #2 aftermath — AA/KK/AKs folded ~100%).  These are coarse
# health bounds, not a strength eval — a checkpoint can clear them and still play
# poorly.  Tuned to fail the known all-in collapse (preflop all-in ~28%, PFR ~53%,
# avg raise ~37x, strong all-in ~54%) AND the known fold collapse (strong continue
# ~0%) while passing a non-collapsed preflop strategy.
UNHEALTHY_THRESHOLDS = {
    "preflop_all_in": 8.0,    # % of preflop random spots that shove
    "preflop_pfr": 40.0,      # % preflop raise frequency
    "preflop_avg_raise": 10.0,  # avg preflop bet/raise in pots
    "strong_all_in": 25.0,    # % of AA/KK/AKs spots that shove
    "strong_continue_min": 80.0,  # % of AA/KK/AKs spots that continue (low=bad)
}


def _avg_size(stats):
    return sum(stats["sizes"]) / len(stats["sizes"]) if stats["sizes"] else 0.0


def _strong_continue_pct(strong):
    """% of strong-hand spots that continue (check/call/bet/raise/all_in)."""
    return _pct(strong["total"] - strong["fold"], strong["total"])


def _evaluate_health(pre, strong, thresholds=None):
    """Return [(label, value, threshold, direction, tripped)] for the gate.

    ``direction`` is ``"high"`` (unhealthy when value >= threshold, the
    all-in-collapse signature) or ``"low"`` (unhealthy when value < threshold,
    the fold-collapse signature).  ``tripped`` applies that comparison.
    """
    t = thresholds or UNHEALTHY_THRESHOLDS
    metrics = [
        ("preflop all-in %", _pct(pre["all_in"], pre["total"]),
         t["preflop_all_in"], "high"),
        ("preflop PFR %", _pct(pre["pfr"], pre["total"]),
         t["preflop_pfr"], "high"),
        ("avg preflop raise (x pot)", _avg_size(pre),
         t["preflop_avg_raise"], "high"),
        ("strong-hand all-in %", _pct(strong["all_in"], strong["total"]),
         t["strong_all_in"], "high"),
        ("strong-hand continue %", _strong_continue_pct(strong),
         t["strong_continue_min"], "low"),
    ]
    rows = []
    for label, value, thr, direction in metrics:
        tripped = value >= thr if direction == "high" else value < thr
        rows.append((label, value, thr, direction, tripped))
    return rows


def _print_health_gate(pre, strong):
    """Print the failure-mode gate and return True if healthy (no metric tripped)."""
    rows = _evaluate_health(pre, strong)
    print("=" * 72)
    print("HEALTH GATE (--fail-on-unhealthy)")
    print("=" * 72)
    tripped_any = False
    for label, value, thr, direction, tripped in rows:
        mark = "FAIL" if tripped else "OK"
        # ``op`` describes value-vs-threshold so the reader sees why it tripped.
        if direction == "high":
            op = ">=" if tripped else "<"
        else:
            op = "<" if tripped else ">="
        print(f"  {mark}: {label:<28} {value:7.2f}  ({op} {thr:.1f})")
        tripped_any = tripped_any or tripped
    healthy = not tripped_any
    print("-" * 72)
    print(f"  RESULT: {'HEALTHY' if healthy else 'UNHEALTHY'}")
    print()
    return healthy


def run_probe(args):
    random.seed(args.seed)
    bot = _load_bot(args.weights, args.disable_search)

    trace_n = max(0, int(args.trace))

    pre = _new_stats()
    pre_traces = []
    for i in range(args.samples):
        hole = _random_hole()
        view = _view(
            hole,
            position=random.choice(POSITIONS),
            pot=15,
            to_call=10,
            stack=args.chips,
        )
        # Swapping act()->act_with_trace() for the first N samples does not
        # perturb the RNG stream (the trace path draws no extra randomness), so
        # distribution stats are identical to a --trace 0 run.
        if i < trace_n:
            action, trace = bot.act_with_trace(view)
            pre_traces.append((view, trace))
        else:
            action = bot.act(view)
        _record(pre, action, view)

    post = _new_stats()
    for _ in range(args.samples):
        hole = _random_hole()
        board = _random_board(hole, 3)
        pot = random.choice([30, 50, 80, 120, 200])
        to_call = random.choice([0, 10, 20, 40, 80])
        to_call = min(to_call, pot)
        view = _view(
            hole,
            board=board,
            street="flop",
            position=random.choice(POSITIONS),
            pot=pot,
            to_call=to_call,
            stack=args.chips,
        )
        action = bot.act(view)
        _record(post, action, view)

    strong = _new_stats()
    strong_traces = []
    strong_hands = [
        [("A", "h"), ("A", "s")],
        [("K", "h"), ("K", "s")],
        [("A", "h"), ("K", "h")],
    ]
    for i in range(50):
        view = _view(
            strong_hands[i % len(strong_hands)],
            position=random.choice(["CO", "BTN"]),
            pot=15,
            to_call=10,
            stack=args.chips,
        )
        if i < trace_n:
            action, trace = bot.act_with_trace(view)
            strong_traces.append((view, trace))
        else:
            action = bot.act(view)
        _record(strong, action, view)

    trash = _new_stats()
    trash_hands = [
        [("7", "h"), ("2", "c")],
        [("3", "h"), ("2", "c")],
    ]
    for i in range(50):
        view = _view(
            trash_hands[i % len(trash_hands)],
            position=random.choice(["UTG", "MP"]),
            pot=15,
            to_call=10,
            stack=args.chips,
        )
        action = bot.act(view)
        _record(trash, action, view)

    print("=" * 72)
    print("Deep CFR Action-Distribution Probe")
    print("=" * 72)
    print(f"Weights:        {args.weights}")
    print(f"Search:         {'disabled' if args.disable_search else 'enabled'}")
    print(f"Samples:        {args.samples} random preflop, {args.samples} random flop")
    print(f"Seed:           {args.seed}")
    print("=" * 72)
    print()

    _print_stats("A. Preflop random spots", pre, healthy=_preflop_health)
    _print_stats("B. Postflop random flop spots", post)

    strong_hand_continue_pct = _strong_continue_pct(strong)
    _print_stats("C. Strong-hand sanity (AA/KK/AKs late position)", strong)
    print(f"strong_hand_continue_pct: {strong_hand_continue_pct:5.1f}%  "
          f"(continue = check/call/bet/raise/all_in)")
    print(("OK" if strong_hand_continue_pct >= 80 else "FLAG") +
          ": target is >=80% continuing strong hands (below = fold collapse)")
    print()

    trash_fold = _pct(trash["fold"], trash["total"])
    _print_stats("D. Trash-hand sanity (72o/32o early position)", trash)
    print(f"Trash-hand fold rate: {trash_fold:5.1f}%")
    print(("OK" if trash_fold >= 70 else "FLAG") +
          ": target is >70% folding trash early")
    print()

    if trace_n and (pre_traces or strong_traces):
        print("=" * 72)
        print(f"E. Decision traces (first {trace_n} of each group)")
        print("=" * 72)
        print("Legend: an EFFECTIVE ALL-IN line distinguishes whether the")
        print("max-stack raise came from the abstract 'all_in' row or from a")
        print("normal bucket (e.g. bet_100) that converted to a max raise.")
        print()
        if pre_traces:
            print("Preflop random spots")
            print("-" * 40)
            for n, (view, trace) in enumerate(pre_traces, 1):
                _print_trace("preflop", n, view, trace)
        if strong_traces:
            print("Strong-hand sanity (AA/KK/AKs)")
            print("-" * 40)
            for n, (view, trace) in enumerate(strong_traces, 1):
                _print_trace("strong", n, view, trace)

    # Failure-mode gate is additive: only runs when --fail-on-unhealthy is
    # passed, so default probe output and exit code (0) are unchanged.
    if getattr(args, "fail_on_unhealthy", False):
        healthy = _print_health_gate(pre, strong)
        return 0 if healthy else 1
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe a Deep CFR checkpoint's action distribution."
    )
    parser.add_argument("--weights", default="models/deep_cfr_v1.pt")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--chips", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-search", action="store_true")
    parser.add_argument(
        "--trace", type=int, default=0, metavar="N",
        help="Print detailed decision traces for the first N preflop random "
             "samples and the first N strong-hand sanity samples.",
    )
    parser.add_argument(
        "--fail-on-unhealthy", action="store_true",
        help="Exit nonzero if preflop all-in >= 8%%, preflop PFR >= 40%%, avg "
             "preflop raise >= 10x pot, strong-hand all-in >= 25%%, or "
             "strong-hand continue < 80%% (fold collapse). Prints a HEALTH GATE "
             "section; normal probe output is otherwise unchanged.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run_probe(parse_args()))
