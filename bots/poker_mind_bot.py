# bots/poker_mind_bot.py

from core.bot_api import Action, PlayerView
from core.engine import eval_hand


class SmartBot:
    """
    Updated SmartBot compatible with PlayerView object.
    Uses simple heuristic poker logic for preflop and postflop.
    """

    def act(self, state: PlayerView):
        """
        Main decision function.
        'state' is a PlayerView, NOT a dict.
        """
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        stacks = state.stacks
        street = state.street
        position = state.position

        # ===========================
        #        PREFLOP LOGIC
        # ===========================
        if street == "preflop":
            ranks = sorted([c[0] for c in hole], reverse=True)
            r1, r2 = ranks
            
            # Position awareness: play tighter in early position
            position_tightness = self._get_position_tightness(position)
            
            # Premium hands
            if (r1 == "A" and r2 in ("K", "Q", "J")) or (r1 == r2 and r1 in ("A", "K", "Q", "J")):
                return self._raise_or_call(legal, pot)
            
            # Medium-strength hands - adjust by position
            if r1 in ("A", "K", "Q", "J"):
                if position_tightness > 0.5:  # Early position
                    return self._call_or_check(legal)
                else:  # Late position
                    return self._raise_or_call(legal, pot)
            
            # Trash hands - fold in early position, check in late
            if position_tightness > 0.5:
                return self._fold_or_check(legal, to_call)
            else:
                return self._fold_or_check(legal, to_call)

        # ===========================
        #        POSTFLOP LOGIC
        # ===========================
        strength = self._approx_strength(hole, board)
        
        # Position awareness: bet more aggressively in late position
        position_tightness = self._get_position_tightness(position)
        adjusted_strength = strength * (1.0 + (1.0 - position_tightness) * 0.1)  # Boost in late position

        if adjusted_strength >= 0.75:
            return self._raise_or_call(legal, pot)

        if adjusted_strength >= 0.40:
            return self._call_or_check(legal)

        return self._fold_or_check(legal, to_call)

    def _get_position_tightness(self, position):
        """
        Returns position tightness factor: 1.0 = early (tight), 0.0 = late (loose)
        """
        position_order = {
            "UTG": 1.0, "UTG+1": 0.9, "MP": 0.7, "LJ": 0.6,
            "HJ": 0.4, "CO": 0.2, "BTN": 0.0, "SB": 0.3, "BB": 0.5
        }
        return position_order.get(position, 0.5)

    # -----------------------------------------------------
    # HELPER DECISION FUNCTIONS
    # -----------------------------------------------------

    def _fold_or_check(self, legal, to_call):
        """Check if possible, otherwise fold."""
        if to_call > 0:
            return Action("fold")

        # check if allowed
        for a in legal:
            if a["type"] == "check":
                return Action("check")

        return Action("fold")

    def _call_or_check(self, legal):
        """Call if facing a bet, otherwise check."""
        for a in legal:
            if a["type"] == "call":
                return Action("call")

        for a in legal:
            if a["type"] == "check":
                return Action("check")

        return Action("fold")

    def _raise_or_call(self, legal, pot):
        """Raise about 50% pot if possible, else call. Cap at 25% of stack."""
        for a in legal:
            if a["type"] in ("raise", "bet"):
                # Cap raise to avoid all-in escalation on marginal spots
                stack_cap = a["max"] * 0.25
                size = max(a["min"], min(a["max"], pot * 0.5, stack_cap))
                # Don't commit more than half our stack on a single raise
                if size > a["max"] * 0.5:
                    return Action("call") if any(x["type"] == "call" for x in legal) else Action(a["type"], a["min"])
                return Action(a["type"], int(size))

        # fallback to call
        for a in legal:
            if a["type"] == "call":
                return Action("call")

        # fallback to check
        for a in legal:
            if a["type"] == "check":
                return Action("check")

        return Action("fold")

    # -----------------------------------------------------
    # Hand strength estimator (FIXED)
    # -----------------------------------------------------
    def _approx_strength(self, hole, board):
        """Calculate hand strength using the treys evaluator (0.0–1.0)."""
        if not hole or len(hole) < 2:
            return 0.0
        score = eval_hand(hole, board)
        return min(1.0, score / 7462.0)


# backward compatibility
PokerMindBot = SmartBot