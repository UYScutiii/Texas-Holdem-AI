# bots/icm_bot.py
"""
ICMBot — Tournament-aware poker bot using Independent Chip Model (ICM).

Instead of maximising raw chip EV, ICMBot converts stacks into tournament
equity (prize shares) and only takes actions whose ICM-adjusted EV is
positive.  Key behaviours:

  * Large stack → aggressive: ICM equity is already high, so the bot can
    absorb variance and pressure short stacks.
  * Short stacks near elimination → tighten up: the marginal equity gain
    from busting an opponent is already "free", so ICMBot avoids marginal
    spots that risk its own survival.
  * Pot odds are compared against ICM-weighted equity, not raw chip odds.
"""

import random
import math
from itertools import combinations
from core.bot_api import Action, PlayerView
from core.engine import eval_hand, EVAL_HAND_MAX, _FULL_DECK


# ─────────────────────────────────────────────────────────────────────────────
# ICM Equity Calculator
# ─────────────────────────────────────────────────────────────────────────────

def icm_equity(stacks: dict[str, int]) -> dict[str, float]:
    """
    Compute each player's Independent Chip Model equity as a fraction
    of the total prize pool (normalised to 1.0).

    Uses the Malmuth-Harville model: the probability a player finishes in
    a given position is proportional to their stack / remaining stacks,
    iterated recursively for each finishing position.

    For a flat payout structure (everyone gets the same share), ICM equity
    equals stack fraction.  For a typical tournament with top-heavy payouts,
    short stacks have *more* equity per chip than big stacks.

    We use a flat payout here (every place pays 1/N equally) so that the
    ICM pressure curve is entirely driven by stack-size risk aversion —
    big stacks can afford to gamble, short stacks cannot.

    Returns:
        Dict mapping player_id → equity ∈ [0, 1].
    """
    players = [pid for pid, s in stacks.items() if s > 0]
    n = len(players)
    if n == 0:
        return {pid: 0.0 for pid in stacks}
    if n == 1:
        return {pid: (1.0 if stacks[pid] > 0 else 0.0) for pid in stacks}

    total = sum(stacks[p] for p in players)
    if total == 0:
        eq = 1.0 / n
        return {pid: (eq if pid in players else 0.0) for pid in stacks}

    # Payout vector: equal shares for simplicity (pure ICM pressure).
    payouts = [1.0 / n] * n

    # Compute probabilities of finishing in each position via recursion.
    equity = {pid: 0.0 for pid in stacks}

    def _recurse(remaining: list[str], remaining_total: int, payout_idx: int):
        """
        For each remaining player, compute their probability of finishing
        in position `payout_idx`, multiply by that position's payout, and
        recurse for the remaining positions.
        """
        if payout_idx >= len(payouts) or not remaining:
            return
        if len(remaining) == 1:
            # Last player gets all remaining payouts
            pid = remaining[0]
            for i in range(payout_idx, len(payouts)):
                equity[pid] += payouts[i]
            return

        for pid in remaining:
            prob = stacks[pid] / remaining_total if remaining_total > 0 else 1.0 / len(remaining)
            equity[pid] += prob * payouts[payout_idx]

            # Recurse: remove this player, compute rest
            new_remaining = [p for p in remaining if p != pid]
            new_total = remaining_total - stacks[pid]
            _recurse(new_remaining, new_total, payout_idx + 1)

    # Limit recursion depth for large fields (exact ICM is O(N!) but for
    # typical 6-9 player tables it's fine).
    if n <= 8:
        _recurse(players, total, 0)
    else:
        # Approximation for huge tables: equity ≈ stack fraction
        for pid in players:
            equity[pid] = stacks[pid] / total

    return equity


def icm_ev_of_call(
    my_pid: str,
    stacks: dict[str, int],
    pot: int,
    to_call: int,
    win_prob: float,
) -> float:
    """
    Compute the ICM-adjusted EV of calling a bet.

    Compares the ICM equity in two scenarios:
      * WIN:  hero's stack increases by (pot + to_call - to_call) = pot
      * LOSE: hero's stack decreases by to_call

    Returns the expected change in ICM equity (positive = profitable).
    """
    my_stack = stacks.get(my_pid, 0)
    if my_stack <= 0:
        return -1.0  # Already busted

    current_eq = icm_equity(stacks)
    my_current = current_eq.get(my_pid, 0.0)

    # Scenario: we call and WIN
    stacks_win = dict(stacks)
    stacks_win[my_pid] = my_stack + pot  # we gain the whole pot
    # The opponent who was betting loses their contribution (already in pot)
    # For simplicity, we just adjust our stack — other stacks stay the same
    eq_win = icm_equity(stacks_win).get(my_pid, 0.0)

    # Scenario: we call and LOSE
    stacks_lose = dict(stacks)
    stacks_lose[my_pid] = max(0, my_stack - to_call)
    eq_lose = icm_equity(stacks_lose).get(my_pid, 0.0)

    expected_eq = win_prob * eq_win + (1 - win_prob) * eq_lose
    return expected_eq - my_current


# ─────────────────────────────────────────────────────────────────────────────
# ICMBot
# ─────────────────────────────────────────────────────────────────────────────

class ICMBot:
    """
    Tournament-aware bot that uses ICM equity to adjust decisions.

    Design principles:
      1. Estimate hand equity via Monte Carlo rollout (same as MonteCarloBot).
      2. Convert pot odds from chip-EV to ICM-EV using the ICM calculator.
      3. Use ICM "pressure" (how much equity we risk vs. gain) to widen or
         narrow our playing range dynamically.
    """

    def __init__(self, simulations: int = 300):
        self.simulations = simulations

    # ── Public interface ──────────────────────────────────────────────────

    def act(self, state: PlayerView) -> Action:
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        stacks = state.stacks
        street = state.street
        position = state.position
        my_pid = state.me
        opponents = state.opponents

        # If no hole cards (already folded / sitting out)
        if not hole:
            return self._choose("check", legal)

        my_stack = stacks.get(my_pid, 0)
        total_chips = sum(s for s in stacks.values() if s > 0)
        n_players = len([s for s in stacks.values() if s > 0])

        # ── Compute hand equity via Monte Carlo ──────────────────────
        n_opps = max(1, len(opponents))
        win_prob = self._estimate_equity(hole, board, n_opps)

        # ── ICM pressure metrics ─────────────────────────────────────
        stack_ratio = my_stack / total_chips if total_chips > 0 else 0.5
        avg_stack = total_chips / n_players if n_players > 0 else my_stack

        # Am I a big stack? (> 1.5× average)
        is_big_stack = my_stack > avg_stack * 1.5
        # Am I short? (< 0.6× average)
        is_short_stack = my_stack < avg_stack * 0.6

        # Are any opponents critically short? (< 0.3× average)
        short_opponents = sum(
            1 for pid in opponents
            if 0 < stacks.get(pid, 0) < avg_stack * 0.3
        )
        opponents_near_bust = short_opponents > 0

        # Position tightness factor (1.0 = early/tight, 0.0 = late/loose)
        pos_tightness = self._get_position_tightness(position)

        # ── ICM-adjusted thresholds ──────────────────────────────────
        # Base thresholds
        call_threshold = 0.0   # min equity to call (above pot odds)
        bet_threshold = 0.60   # min equity to bet
        raise_threshold = 0.65 # min equity to raise

        # ICM adjustments:
        # 1. When opponents are near bust, tighten up to let them bust
        #    (we gain equity for free when they bust out).
        if opponents_near_bust and not is_big_stack:
            call_threshold += 0.08
            bet_threshold += 0.10
            raise_threshold += 0.10

        # 2. Big stack → play more aggressively to pressure short stacks
        if is_big_stack:
            call_threshold -= 0.05
            bet_threshold -= 0.08
            raise_threshold -= 0.05

        # 3. Short stack → tighten to survive, but shove when strong
        if is_short_stack:
            call_threshold += 0.06
            bet_threshold += 0.05
            # But if we're desperate (< 5 BBs effective), widen for shove
            effective_bbs = my_stack / max(1, pot // 3) if pot > 0 else 10
            if effective_bbs < 5:
                bet_threshold -= 0.15  # push-or-fold territory

        # 4. Position adjustment
        if pos_tightness > 0.5:  # Early position
            bet_threshold += 0.05
            raise_threshold += 0.05
        else:  # Late position
            bet_threshold -= 0.05
            raise_threshold -= 0.05

        # ── Decision logic ───────────────────────────────────────────

        # FACING A BET
        if to_call > 0:
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0

            # Compute ICM EV of calling
            icm_delta = icm_ev_of_call(my_pid, stacks, pot, to_call, win_prob)

            # ICM says fold: the call loses equity even if chip-EV is close
            if icm_delta < -0.005 and win_prob < pot_odds + call_threshold:
                return self._choose("fold", legal)

            # Big call relative to stack — need strong hand or positive ICM
            if my_stack > 0 and to_call > my_stack * 0.4:
                if win_prob < 0.65 and icm_delta < 0.01:
                    return self._choose("fold", legal)

            # Equity doesn't beat pot odds + threshold
            if win_prob < pot_odds + call_threshold:
                return self._choose("fold", legal)

            # Strong enough to raise
            if win_prob >= raise_threshold and icm_delta > 0.005:
                return self._try_raise(pot, legal, stacks, my_pid, is_big_stack)

            # Good enough to call
            return self._choose("call", legal)

        # NOT FACING A BET
        if win_prob >= bet_threshold:
            return self._try_bet(pot, legal, stacks, my_pid, is_big_stack)

        # Medium hand → check
        return self._choose("check", legal)

    # ── Monte Carlo equity estimation ────────────────────────────────────

    def _estimate_equity(self, hole, board, num_opponents=1):
        """Estimate win probability via Monte Carlo rollout."""
        wins = 0
        ties = 0
        sims = self.simulations

        # Build the used-card set and remaining deck ONCE before the loop.
        base_used = set(tuple(c) for c in hole) | set(tuple(c) for c in board)
        base_remaining = [c for c in _FULL_DECK if c not in base_used]
        need_board = 5 - len(board)

        for _ in range(sims):
            sim_used = base_used.copy()
            opp_hands = []
            valid = True

            for _ in range(num_opponents):
                avail = [c for c in base_remaining if c not in sim_used]
                if len(avail) < 2:
                    valid = False
                    break
                opp = random.sample(avail, 2)
                opp_hands.append(opp)
                sim_used |= {tuple(c) for c in opp}

            if not valid:
                continue

            if need_board > 0:
                avail_board = [c for c in base_remaining if c not in sim_used]
                if len(avail_board) < need_board:
                    continue
                full_board = list(board) + random.sample(avail_board, need_board)
            else:
                full_board = list(board)

            my_score = eval_hand(hole, full_board)

            opp_scores = [eval_hand(opp, full_board) for opp in opp_hands]
            best_opp = max(opp_scores)

            if my_score > best_opp:
                wins += 1
            elif my_score == best_opp:
                ties += 1

        return (wins + ties * 0.5) / sims

    # ── Sizing helpers ───────────────────────────────────────────────────

    def _try_bet(self, pot, legal, stacks, my_pid, is_big_stack):
        """Attempt a bet, sized according to ICM pressure."""
        for a in legal:
            if a["type"] == "bet":
                if is_big_stack:
                    # Bigger bets to pressure opponents
                    target = pot * 0.65
                    stack_cap = a["max"] * 0.35
                else:
                    # Smaller, conservative bets
                    target = pot * 0.40
                    stack_cap = a["max"] * 0.20
                amt = max(a["min"], min(a["max"], int(target), int(stack_cap)))
                return Action("bet", amt)
        return self._choose("check", legal)

    def _try_raise(self, pot, legal, stacks, my_pid, is_big_stack):
        """Attempt a raise, sized according to ICM pressure."""
        for a in legal:
            if a["type"] == "raise":
                if is_big_stack:
                    target = pot * 0.80
                    stack_cap = a["max"] * 0.35
                else:
                    target = pot * 0.60
                    stack_cap = a["max"] * 0.25
                amt = max(a["min"], min(a["max"], int(target), int(stack_cap)))
                return Action("raise", amt)
        return self._choose("call", legal)

    # ── Card helpers ─────────────────────────────────────────────────────

    def _random_hand(self, used):
        deck = self._remaining_deck(used)
        return random.sample(deck, 2)

    def _complete_board(self, board, used):
        deck = self._remaining_deck(used)
        need = 5 - len(board)
        if need <= 0:
            return board
        return board + random.sample(deck, need)

    def _remaining_deck(self, used):
        used_set = set(tuple(c) for c in used)
        return [c for c in _FULL_DECK if c not in used_set]

    # ── Action helpers ───────────────────────────────────────────────────

    def _choose(self, typ, legal):
        """Pick an action type from legal actions, with fallback."""
        for a in legal:
            if a["type"] == typ:
                return Action(typ)
        # Fallback chain
        for a in legal:
            if a["type"] in ("call", "check"):
                return Action(a["type"])
        return Action("fold")

    def _get_position_tightness(self, position):
        """Returns 1.0 = early (tight), 0.0 = late (loose)."""
        position_order = {
            "UTG": 1.0, "UTG+1": 0.9, "MP": 0.7, "LJ": 0.6,
            "HJ": 0.4, "CO": 0.2, "BTN": 0.0, "SB": 0.5, "BB": 0.7
        }
        return position_order.get(position, 0.5)
