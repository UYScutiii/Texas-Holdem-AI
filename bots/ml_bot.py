import torch
from core.bot_api import Action
from core.engine import eval_hand, EVAL_HAND_MAX
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
    def __init__(self, model_path="models/ml_model.pt", device="cpu",
                 use_fallback=True, temperature=0.8, training_mode=False,
                 starting_chips=500):
        self.device = device
        self.use_fallback = use_fallback
        self.temperature = temperature
        self.training_mode = training_mode
        self.starting_chips = max(1, starting_chips)  # normalise stack/pot features
        self.model = PokerMLP(input_dim=26, hidden=128, num_classes=6)
        self.model_trained = False

        # SHORT-TERM MEMORY: Track opponent behavior during this tournament
        self.opponent_stats = {}  # opponent_id -> running stats dict
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
        # Normalise monetary values so features share the same [0, ~2] scale.
        scale = self.starting_chips
        pot = float(state.pot) / scale
        to_call = float(state.to_call) / scale
        hero_stack = float(state.stacks.get(state.me, 0)) / scale
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

        # Hand strength estimate
        hand_strength = self._estimate_hand_strength(hole, board)

        # Pot odds
        if pot + to_call > 0:
            pot_odds = to_call / (pot + to_call)
        else:
            pot_odds = 0.0

        # Position encoding (0.0 = early/tight, 1.0 = late/loose)
        position_order = {
            "UTG": 0.0, "UTG+1": 0.1, "MP": 0.3, "LJ": 0.4,
            "HJ": 0.6, "CO": 0.8, "BTN": 1.0, "SB": 0.5, "BB": 0.3
        }
        position_value = position_order.get(state.position, 0.5)

        # Calculate memory features from running opponent stats
        opponents = state.opponents
        memory_features = self._calculate_memory_features(opponents)

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
        """Hand strength estimate using eval_hand, normalised to [0.0, 1.0]."""
        if not hole or len(hole) < 2:
            return 0.0
        score = eval_hand(hole, board)
        return score / EVAL_HAND_MAX

    # ----------------------------------------------------------
    # ACT ------------------------------------------------------
    # ----------------------------------------------------------
    def act(self, state):
        state = _as_view(state)
        legal = state.legal_actions

        # Update running opponent stats from history
        if hasattr(state, 'history') and state.history:
            self._update_opponent_stats(state.history, state.opponents)

        # If model not trained and fallback enabled, use fallback
        if not self.model_trained and self.use_fallback:
            return self._fallback_strategy(state)

        # Build 26-dim features
        x = self._make_features(state)

        # Predict class
        logits = self.model(x)

        # Temperature-scaled softmax for exploration (non-training mode)
        if not self.training_mode and self.temperature != 1.0:
            scaled_logits = logits / self.temperature
        else:
            scaled_logits = logits

        probs = torch.softmax(scaled_logits, dim=1)[0]

        if self.training_mode:
            # Training: always argmax
            pred = int(probs.argmax().item())
        else:
            # Inference: sample from temperature-scaled distribution
            pred = int(torch.multinomial(probs, 1).item())

        confidence = float(probs[pred].item())

        # If low confidence and fallback enabled, use fallback
        if self.use_fallback and confidence < 0.5:
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
            return self._raise_fallback(pot, legal)

        # No bet yet
        if strength > 0.60:
            return self._bet_fallback(pot, legal)
        return self._choose("check", legal)

    def _raise_fallback(self, pot, legal):
        """Raise helper for fallback strategy."""
        for a in legal:
            if a["type"] == "raise":
                amt = max(a["min"], min(a["max"], int(pot * 0.75)))
                return Action("raise", amt)
        return self._choose("call", legal)

    def _bet_fallback(self, pot, legal):
        """Bet helper for fallback strategy."""
        for a in legal:
            if a["type"] == "bet":
                amt = max(a["min"], min(a["max"], int(pot * 0.5)))
                return Action("bet", amt)
        return self._choose("check", legal)

    def _calculate_memory_features(self, opponents):
        """
        Calculate opponent behavior features from running stats.
        Returns [avg_aggression, avg_tightness, avg_vpip].
        Uses all accumulated stats across hands, not just last N actions.
        """
        if not opponents or not self.opponent_stats:
            return [0.5, 0.5, 0.5]  # Neutral values

        total_aggression = 0.0
        total_tightness = 0.0
        total_vpip = 0.0
        counted = 0

        for opp_id in opponents:
            if opp_id in self.opponent_stats:
                stats = self.opponent_stats[opp_id]
                total = stats['action_count']
                if total > 0:
                    total_aggression += stats['aggressive_count'] / total
                    total_tightness += stats['fold_count'] / total
                    total_vpip += stats['vpip_count'] / total
                    counted += 1

        if counted == 0:
            return [0.5, 0.5, 0.5]

        return [
            total_aggression / counted,
            total_tightness / counted,
            total_vpip / counted,
        ]

    def _update_opponent_stats(self, history, opponents):
        """
        Update running opponent stats from current hand's history.
        Tracks cumulative stats across all hands with deduplication.
        Engine history entries use top-level keys:
          {"pid": player_id, "type": action_type, "street": ..., ...}
        """
        if not history or not opponents:
            return

        for entry in history:
            if not isinstance(entry, dict):
                continue
            player = entry.get("pid")
            action_type = entry.get("type", "fold")
            if player not in opponents:
                continue

            if player not in self.opponent_stats:
                self.opponent_stats[player] = {
                    'action_count': 0,
                    'aggressive_count': 0,
                    'fold_count': 0,
                    'vpip_count': 0,
                    '_seen_entries': set(),
                }

            # Deduplicate: avoid re-counting entries we've already processed
            entry_key = (player, action_type, entry.get("street", ""), entry.get("seq", id(entry)))
            stats = self.opponent_stats[player]
            if entry_key in stats['_seen_entries']:
                continue
            stats['_seen_entries'].add(entry_key)

            stats['action_count'] += 1

            if action_type in ("bet", "raise"):
                stats['aggressive_count'] += 1
            if action_type == "fold":
                stats['fold_count'] += 1
            if action_type in ("call", "bet", "raise", "check"):
                stats['vpip_count'] += 1
