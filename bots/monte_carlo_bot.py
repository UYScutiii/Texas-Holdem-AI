import random
from core.bot_api import Action, PlayerView
from core.engine import eval_hand


class MonteCarloBot:
    """
    Monte Carlo rollout bot updated to work with PlayerView objects.
    """

    def __init__(self, simulations=200):
        self.simulations = simulations

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

        # Monte Carlo equity estimate
        winrate = self._estimate_equity(hole, board, sims=self.simulations)
        
        # Position awareness: adjust thresholds based on position
        position_tightness = self._get_position_tightness(position)
        
        # Adjust winrate thresholds: play tighter in early position
        # Early position: need higher winrate to bet/raise
        # Late position: can bet/raise with lower winrate
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
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
            my_stack = state.stacks.get(state.me, 0)

            # Weak hand → fold
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
    # Monte Carlo equity estimation
    # ----------------------------------------------------
    def _estimate_equity(self, hole, board, sims=200):
        wins = 0
        ties = 0

        for _ in range(sims):
            opp = self._random_opponent_hand(hole, board)
            full_board = self._random_board(board, hole, opp)

            my_score = eval_hand(hole, full_board)
            opp_score = eval_hand(opp, full_board)

            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1

        return (wins + ties * 0.5) / sims

    # ----------------------------------------------------
    # Randomize opponent hole cards
    # ----------------------------------------------------
    def _random_opponent_hand(self, hole, board):
        deck = self._remaining_deck(hole, board)
        return random.sample(deck, 2)

    # ----------------------------------------------------
    # Complete the board to 5 cards randomly
    # ----------------------------------------------------
    def _random_board(self, board, hole, opp):
        deck = self._remaining_deck(hole, board, opp)
        need = 5 - len(board)
        cards = random.sample(deck, need)
        return board + cards

    # ----------------------------------------------------
    # Remaining deck helper
    # ----------------------------------------------------
    def _remaining_deck(self, hole, board, opp=None):
        ranks = "23456789TJQKA"
        suits = "cdhs"

        full = [(r, s) for r in ranks for s in suits]

        used = set(tuple(c) for c in hole)
        used |= set(tuple(c) for c in board)
        if opp:
            used |= set(tuple(c) for c in opp)

        return [c for c in full if c not in used]

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
            "HJ": 0.4, "CO": 0.2, "BTN": 0.0, "SB": 0.3, "BB": 0.5
        }
        return position_order.get(position, 0.5)


# Backwards alias
SmartBot = MonteCarloBot