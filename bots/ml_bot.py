import torch
import random
from core.bot_api import Action
from core.engine import eval_hand
from models.poker_mlp import PokerMLP



# CARD ENCODING ------------------------------------------------------

RANKS = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
         "9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
SUITS = {"c":0, "d":1, "h":2, "s":3}

def encode_card(card):
    rank, suit = card
    return [RANKS[rank], SUITS[suit]]


STREET_MAP = {"preflop":0, "flop":1, "turn":2, "river":3}

def _as_view(state):
    """Convert a dict state to an attribute-accessible object if needed."""
    if isinstance(state, dict):
        class _DictView:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        return _DictView(state)
    return state


# ML BOT -------------------------------------------------------------

class MLBot:
    def __init__(self, model_path="models/ml_model.pt", device="cpu", use_fallback=True):
        self.device = device
        self.use_fallback = use_fallback
        self.model = PokerMLP(input_dim=26, hidden=128, num_classes=6)  # Changed from 23 to 26
        self.model_trained = False
        
        # SHORT-TERM MEMORY: Track opponent behavior during this tournament
        self.opponent_stats = {}  # opponent_id -> stats dict
        self.hand_history = []  # Recent hands in this tournament
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Check if model dimensions match
            first_layer_weight = checkpoint.get('net.0.weight', None)
            if first_layer_weight is not None:
                expected_input_dim = first_layer_weight.shape[1]
                if expected_input_dim == 20:
                    print(f"Warning: Model file has old 20-feature format. Need to retrain with 26 features.")
                    print("Using fallback strategy until model is retrained.")
                    self.model_trained = False
                elif expected_input_dim == 23:
                    print(f"Warning: Model file has 23-feature format. Need to retrain with 26 features (includes memory).")
                    print("Using fallback strategy until model is retrained.")
                    self.model_trained = False
                elif expected_input_dim == 26:
                    # New model with 26 features - load it
                    self.model.load_state_dict(checkpoint)
                    self.model.eval()
                    self.model_trained = True
                else:
                    print(f"Warning: Model has unexpected input dimension {expected_input_dim}. Using fallback.")
                    self.model_trained = False
            else:
                # Try loading anyway
                self.model.load_state_dict(checkpoint)
                self.model.eval()
                self.model_trained = True
        except (FileNotFoundError, OSError, RuntimeError) as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using untrained model (random weights).")
            self.model.eval()

    def _make_features(self, state):
        """
        Produce feature vector with hand strength, pot odds, and position.
        Now 26 dimensions: 20 original + 3 new features + 3 memory features
        """
        state = _as_view(state)

        street = STREET_MAP.get(state.street, 0)
        pot = float(state.pot)
        to_call = float(state.to_call)
        hero_stack = float(state.stacks.get(state.me, 0))
        eff_stack = min(hero_stack, min(state.stacks.get(pid, hero_stack) for pid in state.opponents))
        n_players = len(state.opponents) + 1

        # Hole cards encoding (pad to 4 numbers)
        hole = state.hole_cards or []
        hole_enc = []
        for i in range(2):
            if i < len(hole):
                hole_enc.extend(encode_card(hole[i]))
            else:
                hole_enc.extend([0, 0])  # Padding

        # Board encoding (pad to 10 numbers for 5 cards)
        board = state.board or []
        board_enc = []
        for i in range(5):
            if i < len(board):
                board_enc.extend(encode_card(board[i]))
            else:
                board_enc.extend([0, 0])  # Padding

        # NEW: Hand strength estimate
        hand_strength = self._estimate_hand_strength(hole, board)

        # NEW: Pot odds
        if pot + to_call > 0:
            pot_odds = to_call / (pot + to_call)
        else:
            pot_odds = 0.0
        
        # Position encoding (0.0 = early/tight, 1.0 = late/loose)
        position_order = {
            "UTG": 0.0, "UTG+1": 0.1, "MP": 0.3, "LJ": 0.4,
            "HJ": 0.6, "CO": 0.8, "BTN": 1.0, "SB": 0.7, "BB": 0.5
        }
        position_value = position_order.get(state.position, 0.5)

        # NEW: Calculate memory features from opponent history
        opponents = state.opponents
        memory_features = self._calculate_memory_features_from_history(
            state.history, me=state.me, opponents=opponents
        )

        # FULL 26-feature vector
        features = (
            [street, pot, to_call, hero_stack, eff_stack, n_players]
            + hole_enc
            + board_enc
            + [hand_strength, pot_odds, position_value]  # 3 advanced features
            + memory_features  # 3 memory features
        )

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return x.to(self.device)

    def _estimate_hand_strength(self, hole, board):
        """Hand strength estimate using the treys evaluator (0.0–1.0)."""
        if not hole or len(hole) < 2:
            return 0.0
        score = eval_hand(hole, board)
        return min(1.0, score / 7462.0)

    # ----------------------------------------------------------
    # ACT ------------------------------------------------------
    # ----------------------------------------------------------
    def act(self, state):
        state = _as_view(state)
        legal = state.legal_actions

        # Update memory from history
        if hasattr(state, 'history') and state.history:
            self._update_memory_from_history(state.history, state.opponents)

        # If model not trained and fallback enabled, use fallback
        if not self.model_trained and self.use_fallback:
            return self._fallback_strategy(state)

        # Build 26-dim features
        x = self._make_features(state)

        # Predict class
        logits = self.model(x)
        pred = int(logits.argmax(dim=1).item())
        
        # Get confidence (softmax probability)
        probs = torch.softmax(logits, dim=1)[0]
        confidence = float(probs[pred].item())

        # If low confidence and fallback enabled, use fallback
        if self.use_fallback and confidence < 0.4:
            return self._fallback_strategy(state)

        # -------- handle buckets --------
        if pred == 0:
            return self._choose("fold", legal)
        if pred == 1:
            return self._choose("check", legal)
        if pred == 2:
            return self._choose("call", legal)

        # RAISE BUCKETS (3,4,5)
        return self._raise_bucket(pred - 3, legal)

    # ----------------------------------------------------------
    def _choose(self, typ, legal):
        for a in legal:
            if a["type"] == typ:
                return Action(typ)
        for fallback in ("call", "check", "fold"):
            for a in legal:
                if a["type"] == fallback:
                    return Action(fallback)
        a = legal[0]
        return Action(a["type"], a.get("min"))

    # ----------------------------------------------------------
    def _raise_bucket(self, bucket, legal):
        raises = [a for a in legal if a["type"] == "raise"]
        bets = [a for a in legal if a["type"] == "bet"]

        if raises:
            a = raises[0]
        elif bets:
            a = bets[0]
        else:
            return self._choose("call", legal)

        lo, hi = a["min"], a["max"]
        amt = lo + (hi - lo) * (bucket / 2)   # 3 buckets

        return Action(a["type"], round(amt, 2))

    def _fallback_strategy(self, state):
        """Fallback to simple hand strength logic when model is untrained."""
        state = _as_view(state)
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions

        if not hole:
            return self._choose("fold", legal)

        strength = self._estimate_hand_strength(hole, board)

        # Facing a bet
        if to_call > 0:
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
            if strength < pot_odds:
                return self._choose("fold", legal)
            if strength < 0.55:
                return self._choose("call", legal)
            return self._raise(pot, legal)

        # No bet yet
        if strength > 0.60:
            return self._bet(pot, legal)
        return self._choose("check", legal)

    def _calculate_memory_features_from_history(self, history, me, opponents):
        """
        Calculate opponent behavior features from the current hand's history.
        Returns [avg_aggression, avg_tightness, avg_vpip]
        Uses the history from the current hand to track opponent actions.
        """
        if not opponents or not history:
            return [0.5, 0.5, 0.5]  # Neutral values
        
        # Extract opponent actions from current hand history
        opponent_actions = []
        for entry in history:
            if isinstance(entry, dict):
                player = entry.get("pid")
                action = entry.get("action", {})
                if player in opponents and isinstance(action, dict):
                    opponent_actions.append({
                        "player": player,
                        "type": action.get("type", "fold")
                    })
        
        # Also use our stored opponent stats from previous hands
        if hasattr(self, 'opponent_stats') and self.opponent_stats:
            for opp_id in opponents:
                if opp_id in self.opponent_stats:
                    stats = self.opponent_stats[opp_id]
                    # Add to opponent_actions for calculation
                    for _ in range(int(stats.get('action_count', 0))):
                        if stats.get('last_action'):
                            opponent_actions.append({
                                "player": opp_id,
                                "type": stats['last_action']
                            })
        
        if not opponent_actions:
            return [0.5, 0.5, 0.5]  # No history yet
        
        # Calculate stats from recent actions (last 10)
        recent = opponent_actions[-10:]
        total_actions = len(recent)
        
        if total_actions == 0:
            return [0.5, 0.5, 0.5]
        
        aggressive_count = sum(1 for d in recent 
                              if d.get("type") in ("bet", "raise"))
        fold_count = sum(1 for d in recent 
                        if d.get("type") == "fold")
        vpip_count = sum(1 for d in recent 
                        if d.get("type") in ("call", "bet", "raise", "check"))
        
        avg_aggression = aggressive_count / total_actions
        avg_tightness = fold_count / total_actions
        avg_vpip = vpip_count / total_actions
        
        return [avg_aggression, avg_tightness, avg_vpip]

    def _calculate_memory_features(self, all_decisions, current_idx, me, opponents, current_file):
        """
        DEPRECATED: Old method that required hand_id and log_file.
        Kept for backwards compatibility but not used.
        """
        # Fallback to history-based calculation
        return [0.5, 0.5, 0.5]

    def _update_memory_from_history(self, history, opponents):
        """
        Update opponent stats from current hand's history.
        """
        if not history or not opponents:
            return
        
        for entry in history:
            if isinstance(entry, dict):
                player = entry.get("pid")
                action = entry.get("action", {})
                if player in opponents and isinstance(action, dict):
                    action_type = action.get("type", "fold")
                    
                    if player not in self.opponent_stats:
                        self.opponent_stats[player] = {
                            'action_count': 0,
                            'aggressive_count': 0,
                            'fold_count': 0,
                            'vpip_count': 0,
                            'last_action': action_type
                        }
                    
                    stats = self.opponent_stats[player]
                    stats['action_count'] += 1
                    stats['last_action'] = action_type
                    
                    if action_type in ("bet", "raise"):
                        stats['aggressive_count'] += 1
                    if action_type == "fold":
                        stats['fold_count'] += 1
                    if action_type in ("call", "bet", "raise", "check"):
                        stats['vpip_count'] += 1