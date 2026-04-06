import random
from core.bot_api import Action, PlayerView
from core.engine import eval_hand, _FULL_DECK


class MonteCarloBot:
    """
    Monte Carlo rollout bot updated to work with PlayerView objects.
    Simulates equity against N opponents in multi-way pots.
    """

    def __init__(self, simulations=500, num_opponents=None):
        self.simulations = simulations
        self.num_opponents = num_opponents  # None = auto-detect from state

    # ----------------------------------------------------
    # PUBLIC: act() for engine compatibility
    # ----------------------------------------------------
    def act(self, state: PlayerView):
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        street = state.street
        opponents = state.opponents
        position = state.position

        # If no hole cards (folded)
        if not hole:
            for a in legal:
                if a["type"] == "check":
                    return Action("check")
            return Action("fold")

        # Determine number of opponents for simulation
        n_opps = self.num_opponents or len(opponents) or 1

        # Monte Carlo equity estimate against N opponents
        winrate = self._estimate_equity(hole, board, n_opps, sims=self.simulations)

        # Pot odds
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0

        # Position awareness: adjust thresholds based on position
        position_tightness = self._get_position_tightness(position)

        if position_tightness > 0.5:  # Early position
            bet_threshold = 0.70
            raise_threshold = 0.65
        else:  # Late position
            bet_threshold = 0.60
            raise_threshold = 0.55

        # -----------------------------
        # FACING A BET
        # -----------------------------
        if to_call > 0:
            my_stack = state.stacks.get(state.me, 0)

            # Weak hand: fold if equity doesn't beat pot odds
            if winrate < pot_odds:
                return self._choose("fold", legal)

            # Large call (>40% of stack) requires stronger hand
            if my_stack > 0 and to_call > my_stack * 0.4 and winrate < 0.70:
                return self._choose("fold", legal)

            # Medium hand → call
            if winrate < raise_threshold:
                return self._choose("call", legal)

            # Strong → raise
            return self._raise(pot, legal)

        # -----------------------------
        # NO BET YET
        # -----------------------------
        if winrate > bet_threshold:
            return self._bet(pot, legal)

        # Medium → check
        for a in legal:
            if a["type"] == "check":
                return Action("check")

        # fallback
        return self._choose("fold", legal)

    # ----------------------------------------------------
    # Monte Carlo equity estimation (multi-opponent)
    # ----------------------------------------------------
    def _estimate_equity(self, hole, board, num_opponents=1, sims=500):
        wins = 0
        ties = 0

        # Build the used-card set and remaining deck ONCE before the loop.
        base_used = set(tuple(c) for c in hole) | set(tuple(c) for c in board)
        base_remaining = [c for c in _FULL_DECK if c not in base_used]
        need_board = 5 - len(board)

        for _ in range(sims):
            # Per-sim exclusion grows as we deal opponent hands.
            sim_used = base_used.copy()
            remaining = base_remaining  # read-only reference; filter inline below
            opp_hands = []
            valid = True

            for _ in range(num_opponents):
                avail = [c for c in remaining if c not in sim_used]
                if len(avail) < 2:
                    valid = False
                    break
                opp = random.sample(avail, 2)
                opp_hands.append(opp)
                sim_used |= {tuple(c) for c in opp}

            if not valid:
                continue

            # Complete the board from what's still available.
            if need_board > 0:
                avail_board = [c for c in base_remaining if c not in sim_used]
                if len(avail_board) < need_board:
                    continue
                full_board = list(board) + random.sample(avail_board, need_board)
            else:
                full_board = list(board)

            my_score = eval_hand(hole, full_board)

            # Hero must beat ALL opponents
            opp_scores = [eval_hand(opp, full_board) for opp in opp_hands]
            best_opp = max(opp_scores)

            if my_score > best_opp:
                wins += 1
            elif my_score == best_opp:
                ties += 1

        return (wins + ties * 0.5) / sims

    # ----------------------------------------------------
    # Remaining deck helper (used outside _estimate_equity)
    # ----------------------------------------------------
    def _remaining_deck(self, used):
        used_set = set(tuple(c) for c in used)
        return [c for c in _FULL_DECK if c not in used_set]

    # ----------------------------------------------------
    # Helper: choose legal action
    # ----------------------------------------------------
    def _choose(self, typ, legal):
        for a in legal:
            if a["type"] == typ:
                return Action(typ)
        # fallback if that action type isn't available
        for a in legal:
            if a["type"] in ("call", "check"):
                return Action(a["type"])
        return Action("fold")

    # ----------------------------------------------------
    # Raise helper
    # ----------------------------------------------------
    def _raise(self, pot, legal):
        for a in legal:
            if a["type"] == "raise":
                # Cap at 30% of stack to avoid all-in escalation
                stack_cap = a["max"] * 0.30
                amt = max(a["min"], min(a["max"], pot * 0.75, stack_cap))
                return Action("raise", int(amt))
        return self._choose("call", legal)

    # ----------------------------------------------------
    # Bet helper
    # ----------------------------------------------------
    def _bet(self, pot, legal):
        for a in legal:
            if a["type"] == "bet":
                stack_cap = a["max"] * 0.25
                amt = max(a["min"], min(a["max"], pot * 0.5, stack_cap))
                return Action("bet", int(amt))
        return self._choose("check", legal)

    def _get_position_tightness(self, position):
        """
        Returns position tightness factor: 1.0 = early (tight), 0.0 = late (loose)
        """
        position_order = {
            "UTG": 1.0, "UTG+1": 0.9, "MP": 0.7, "LJ": 0.6,
            "HJ": 0.4, "CO": 0.2, "BTN": 0.0, "SB": 0.5, "BB": 0.7
        }
        return position_order.get(position, 0.5)
