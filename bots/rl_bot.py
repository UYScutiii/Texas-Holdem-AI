"""
Reinforcement Learning Bot using Policy Gradient (REINFORCE)
Learns optimal strategies through trial and error.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from core.bot_api import Action, PlayerView
from core.engine import eval_hand, EVAL_HAND_MAX

# Card encoding (same as MLBot)
RANKS = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
         "9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
SUITS = {"c":0, "d":1, "h":2, "s":3}

def encode_card(card):
    rank, suit = card
    return [RANKS[rank], SUITS[suit]]

STREET_MAP = {"preflop":0, "flop":1, "turn":2, "river":3}

class PolicyNetwork(nn.Module):
    """Policy network that outputs action probabilities."""
    def __init__(self, input_dim=26, hidden=512):  # Increased from 256 to 512
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 6)  # 6 actions: fold, check, call, raise_small, raise_medium, raise_large
        )
    
    def forward(self, x):
        return self.net(x)


class RLBot:
    """
    Reinforcement Learning Bot using REINFORCE algorithm.
    Learns by playing games and updating policy based on rewards.
    """
    
    def __init__(self, model_path="models/rl_model.pt", device="cpu",
                 learning_rate=1e-4, training_mode=False, exploration_rate=0.1,
                 use_fallback=True, starting_chips=500):
        self.device = device
        self.training_mode = training_mode
        self.exploration_rate = exploration_rate
        self.use_fallback = use_fallback
        self.starting_chips = max(1, starting_chips)  # used to normalise stack/pot features
        self.model_loaded = False  # Track if model loaded successfully
        self.policy_net = PolicyNetwork(input_dim=26, hidden=512).to(device)  # Increased to 512
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Episode tracking for REINFORCE
        self.current_episode = []
        self.episode_rewards = []
        
        # Load existing model if available
        try:
            checkpoint = torch.load(model_path, map_location=device)
            self.policy_net.load_state_dict(checkpoint)
            self.model_loaded = True
            if training_mode:
                print(f"Loaded RL model from {model_path} (training mode)")
            else:
                print(f"Loaded RL model from {model_path}")
        except (FileNotFoundError, OSError, RuntimeError) as e:
            self.model_loaded = False
            if training_mode:
                print(f"No existing RL model found. Starting fresh training.")
            else:
                print(f"Warning: Could not load RL model from {model_path}: {e}")
                if use_fallback:
                    print("Using fallback strategy.")
        
        self.policy_net.eval() if not training_mode else self.policy_net.train()
        
        # Opponent memory
        self.opponent_stats = {}
    
    def _make_features(self, state):
        """Extract 26-dimensional feature vector (same as MLBot)."""
        # Handle both PlayerView and dict
        if isinstance(state, dict):
            class DictView:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            state = DictView(state)
        
        street = STREET_MAP.get(state.street, 0)
        # Normalise monetary values by starting stack so they are on [0, ~2]
        # rather than raw chip counts (0 – 1000+).
        scale = self.starting_chips
        pot = float(state.pot) / scale
        to_call = float(state.to_call) / scale
        hero_stack = float(state.stacks.get(state.me, 0)) / scale
        eff_stack = min(hero_stack, min(state.stacks.get(pid, hero_stack) for pid in state.opponents))
        n_players = len(state.opponents) + 1
        
        # Hole cards encoding
        hole = state.hole_cards or []
        hole_enc = []
        for i in range(2):
            if i < len(hole):
                hole_enc.extend(encode_card(hole[i]))
            else:
                hole_enc.extend([0, 0])
        
        # Board encoding
        board = state.board or []
        board_enc = []
        for i in range(5):
            if i < len(board):
                board_enc.extend(encode_card(board[i]))
            else:
                board_enc.extend([0, 0])
        
        # Hand strength
        hand_strength = self._estimate_hand_strength(hole, board)
        
        # Pot odds
        if pot + to_call > 0:
            pot_odds = to_call / (pot + to_call)
        else:
            pot_odds = 0.0
        
        # Position encoding
        position_order = {
            "UTG": 0.0, "UTG+1": 0.1, "MP": 0.3, "LJ": 0.4,
            "HJ": 0.6, "CO": 0.8, "BTN": 1.0, "SB": 0.5, "BB": 0.5
        }
        position_value = position_order.get(state.position, 0.5)
        
        # Memory features
        memory_features = self._calculate_memory_features(state.history, state.me, state.opponents)
        
        features = (
            [street, pot, to_call, hero_stack, eff_stack, n_players]
            + hole_enc + board_enc
            + [hand_strength, pot_odds, position_value]
            + memory_features
        )
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _estimate_hand_strength(self, hole, board):
        """Hand strength estimate using eval_hand, normalised to [0.0, 1.0]."""
        if not hole or len(hole) < 2:
            return 0.0
        score = eval_hand(hole, board)
        return score / EVAL_HAND_MAX
    
    def _calculate_memory_features(self, history, me, opponents):
        """Calculate opponent behavior features from action history."""
        if not opponents or not history:
            return [0.5, 0.5, 0.5]

        # History entries written by the engine use top-level keys:
        # {"street": ..., "pid": ..., "type": "fold"|"call"|..., "amount": ...}
        opponent_actions = [
            entry
            for entry in history
            if isinstance(entry, dict) and entry.get("pid") in opponents
        ]

        if not opponent_actions:
            return [0.5, 0.5, 0.5]

        recent = opponent_actions[-10:]
        total = len(recent)
        if total == 0:
            return [0.5, 0.5, 0.5]

        aggressive = sum(1 for d in recent if d.get("type") in ("bet", "raise")) / total
        tightness = sum(1 for d in recent if d.get("type") == "fold") / total
        vpip = sum(1 for d in recent if d.get("type") in ("call", "bet", "raise", "check")) / total

        return [aggressive, tightness, vpip]
    
    def _update_memory(self, history, opponents):
        """Update opponent stats from hand history."""
        if not history or not opponents:
            return
        # Engine history entries: {"pid": ..., "type": "fold"|"call"|..., ...}
        for entry in history:
            if not isinstance(entry, dict):
                continue
            player = entry.get("pid")
            action_type = entry.get("type", "fold")
            if player not in opponents:
                continue

            if player not in self.opponent_stats:
                self.opponent_stats[player] = {
                    'action_count': 0, 'aggressive_count': 0,
                    'fold_count': 0, 'vpip_count': 0, 'last_action': action_type
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
    
    def _fallback_strategy(self, state):
        """Fallback to simple hand strength logic when model is not loaded."""
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
            # Raise with strong hands
            return self._raise_bucket(1, legal)  # Medium raise
        
        # No bet yet
        if strength > 0.60:
            return self._bet(pot, legal)
        return self._choose("check", legal)
    
    def _bet(self, pot, legal):
        """Make a bet."""
        for a in legal:
            if a["type"] == "bet":
                amt = max(a["min"], min(a["max"], pot * 0.5))
                return Action("bet", round(amt, 2))
        return self._choose("check", legal)
    
    def act(self, state):
        """Choose action using policy network."""
        try:
            # Handle dict/PlayerView
            if isinstance(state, dict):
                class DictView:
                    def __init__(self, d):
                        for k, v in d.items():
                            setattr(self, k, v)
                state = DictView(state)
            
            legal = state.legal_actions
            
            # Update memory
            if hasattr(state, 'history') and state.history:
                self._update_memory(state.history, state.opponents)
            
            # Use fallback if model not loaded and fallback enabled
            if not self.model_loaded and self.use_fallback and not self.training_mode:
                return self._fallback_strategy(state)
            
            # Get features
            features = self._make_features(state)
            
            # Get action probabilities - CRITICAL: only use no_grad in eval mode
            if self.training_mode:
                # Training mode: NEED gradients for backprop
                logits = self.policy_net(features)
                probs = torch.softmax(logits, dim=1)
            else:
                # Eval mode: no gradients needed
                with torch.no_grad():
                    logits = self.policy_net(features)
                    probs = torch.softmax(logits, dim=1)
            
            # Epsilon-greedy exploration during training
            if self.training_mode and random.random() < self.exploration_rate:
                # Explore: random action - but still need gradients for log_prob
                random_action = random.randint(0, 5)
                action_idx = torch.tensor([random_action], device=self.device)
                # Compute log_prob from distribution (has gradients)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(action_idx)
            elif self.training_mode:
                # Exploit: sample from policy (with gradients)
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)
            else:
                # Eval mode: take best action (no gradients)
                action_idx = probs.argmax(dim=1)
                log_prob = torch.log(probs[0, action_idx] + 1e-8)
            
            # Store for training
            if self.training_mode:
                self.current_episode.append({
                    'state': features,
                    'action': action_idx.item(),
                    'log_prob': log_prob,  # Now has gradients!
                    'legal_actions': legal
                })
            
            # Convert to actual action
            return self._action_idx_to_action(action_idx.item(), legal)
        
        except Exception as e:
            # If anything goes wrong, use fallback
            if self.use_fallback:
                try:
                    return self._fallback_strategy(state)
                except:
                    # Last resort: just fold
                    legal = state.legal_actions if hasattr(state, 'legal_actions') else []
                    return self._choose("fold", legal)
            else:
                raise
    
    def _action_idx_to_action(self, idx, legal):
        """Convert action index to Action object."""
        # Ensure idx is valid
        idx = max(0, min(5, idx))
        
        if idx == 0:  # fold
            return self._choose("fold", legal)
        elif idx == 1:  # check
            return self._choose("check", legal)
        elif idx == 2:  # call
            return self._choose("call", legal)
        else:  # raise buckets (3, 4, 5)
            bucket = min(2, max(0, idx - 3))  # Clamp to 0-2
            return self._raise_bucket(bucket, legal)
    
    def _choose(self, typ, legal):
        """Choose legal action of given type."""
        for a in legal:
            if a["type"] == typ:
                return Action(typ)
        for fallback in ("call", "check", "fold"):
            for a in legal:
                if a["type"] == fallback:
                    return Action(fallback)
        a = legal[0]
        return Action(a["type"], a.get("min"))
    
    def _raise_bucket(self, bucket, legal):
        """Choose raise amount based on bucket (0=small, 1=medium, 2=large)."""
        raises = [a for a in legal if a["type"] == "raise"]
        bets = [a for a in legal if a["type"] == "bet"]
        
        if raises:
            a = raises[0]
        elif bets:
            a = bets[0]
        else:
            return self._choose("call", legal)
        
        lo, hi = a["min"], a["max"]
        amt = lo + (hi - lo) * (bucket / 2.0)
        return Action(a["type"], round(amt, 2))
    
    def record_reward(self, reward):
        """
        Record reward for current episode.
        Call this after each hand with the chip change.
        """
        if self.training_mode and self.current_episode:
            # Store reward for all actions in this episode
            for step in self.current_episode:
                step['reward'] = reward
            self.episode_rewards.append(reward)
    
    def end_episode(self):
        """End current episode and update policy if training."""
        if not self.training_mode or not self.current_episode:
            self.current_episode = []
            return
        
        # Calculate returns (REINFORCE)
        returns = []
        total_return = 0
        
        # Discounted returns (backwards) - use higher discount for tournament play
        for step in reversed(self.current_episode):
            total_return = step['reward'] + 0.95 * total_return  # gamma=0.95 (was 0.99)
            returns.insert(0, total_return)
        
        # Normalize returns
        if returns:
            returns = torch.tensor(returns, dtype=torch.float32)
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            else:
                # If all rewards are same, just use them as-is
                returns = returns - returns.mean()
        
        # Update policy
        self.policy_net.train()
        self.optimizer.zero_grad()
        
        policy_loss = 0
        for step, ret in zip(self.current_episode, returns):
            state = step['state']
            action = step['action']
            log_prob = step['log_prob']
            
            # REINFORCE: -log_prob * return (negative because we minimize)
            policy_loss -= log_prob * ret
        
        if len(self.current_episode) > 0:
            policy_loss = policy_loss / len(self.current_episode)
            policy_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            
            self.optimizer.step()
        
        self.policy_net.eval()
        
        # Reset episode
        self.current_episode = []
    
    def save_model(self, path="models/rl_model.pt"):
        """Save the trained model."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        print(f"RL model saved to {path}")
