"""
Reinforcement Learning Bot using Proximal Policy Optimization (PPO).
Learns optimal strategies through trial and error.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
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
    def __init__(self, input_dim=26, hidden=256):  # Increased from 256 to 512
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


class ValueNetwork(nn.Module):
    """Value network that estimates V(s) for PPO baseline."""
    def __init__(self, input_dim=26, hidden=256):
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
            nn.Linear(hidden // 2, 1),  # single scalar value output
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RLBot:
    """
    Reinforcement Learning Bot using PPO (Proximal Policy Optimization).
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
        self.policy_net = PolicyNetwork(input_dim=26, hidden=512).to(device)
        self.value_net  = ValueNetwork(input_dim=26,  hidden=512).to(device)
        self.optimizer       = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(),  lr=learning_rate)

        # Episode tracking for PPO
        self.current_episode = []
        self.episode_rewards = []
        self._hand_step_start = 0   # index into current_episode where current hand's steps begin

        # Batch buffer: collect multiple episodes before each update
        self.episode_buffer = []
        self.batch_size     = 32

        # Load existing model if available
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # New format: {'policy': ..., 'value': ...}
            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy'])
                self.value_net.load_state_dict(checkpoint['value'])
            else:
                # Old format: bare policy state dict — load policy only
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

        if not training_mode:
            self.policy_net.eval()
            self.value_net.eval()
        else:
            self.policy_net.train()
            self.value_net.train()

        # Opponent memory
        self.opponent_stats = {}
        # Running mean of episode returns (kept for diagnostics)
        self.baseline_returns = deque(maxlen=1000)
    
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
            
            # Get action probabilities
            if self.training_mode:
                # Training mode: need gradients for the PPO update
                logits = self.policy_net(features)
                probs  = torch.softmax(logits, dim=1)
                value  = self.value_net(features)   # shape: (1,)
            else:
                # Eval mode: no gradients needed
                with torch.no_grad():
                    logits = self.policy_net(features)
                    probs  = torch.softmax(logits, dim=1)

            # Epsilon-greedy exploration during training
            if self.training_mode and random.random() < self.exploration_rate:
                # Explore: random action — still track log_prob for PPO ratio
                random_action = random.randint(0, 5)
                action_idx = torch.tensor([random_action], device=self.device)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(action_idx)
            elif self.training_mode:
                # Exploit: sample from policy
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)
            else:
                # Eval mode: greedy
                action_idx = probs.argmax(dim=1)
                log_prob = torch.log(probs[0, action_idx] + 1e-8)

            # Store trajectory step for PPO update
            if self.training_mode:
                self.current_episode.append({
                    'state':    features,
                    'action':   action_idx.item(),
                    'log_prob': log_prob.detach(),   # old π(a|s) — frozen
                    'value':    value.detach(),       # V(s) old estimate — frozen
                    'legal_actions': legal,
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
        Record reward for the most recent hand's steps only.

        Call this after each hand with the normalised chip delta
        ``(final_chips - starting_chips) / starting_chips``.
        Only the steps belonging to the hand just played are tagged;
        earlier hands keep their own rewards.
        """
        if self.training_mode and self.current_episode:
            # Tag only the steps that belong to the hand that just finished
            for step in self.current_episode[self._hand_step_start:]:
                step['reward'] = reward
            # Advance the marker so the next hand's reward won't overwrite these
            self._hand_step_start = len(self.current_episode)
            self.episode_rewards.append(reward)
    
    def end_episode(self):
        """Buffer the completed episode; run PPO update every batch_size episodes."""
        if not self.training_mode or not self.current_episode:
            self.current_episode = []
            self._hand_step_start = 0
            return

        # Move the finished episode into the buffer and reset
        self.episode_buffer.append(list(self.current_episode))
        self.current_episode = []
        self._hand_step_start = 0

        # Only update when we have a full batch
        if len(self.episode_buffer) >= self.batch_size:
            self._ppo_update(self.episode_buffer)
            self.episode_buffer = []

    def flush_buffer(self):
        """Force a PPO update on any remaining buffered episodes (call at end of training)."""
        if self.training_mode and self.episode_buffer:
            self._ppo_update(self.episode_buffer)
            self.episode_buffer = []

    def save_model(self, path="models/rl_model.pt"):
        """Save policy and value network weights to disk."""
        import os
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        torch.save({
            'policy': self.policy_net.state_dict(),
            'value':  self.value_net.state_dict(),
        }, path)

    def _ppo_update(self, episodes):
        """
        Run a PPO update over a batch of completed episodes.

        Uses clipped surrogate objective (PPO-clip) with:
          - GAE-lambda advantage estimation (gamma=0.99, lambda=0.95)
          - Shared normalisation of advantages across the batch
          - Entropy bonus to encourage exploration
          - Gradient clipping for stability

        Args:
            episodes: list of episode buffers, each a list of step dicts
                      with keys: state, action, log_prob, value, reward
        """
        GAMMA     = 0.99
        LAMBDA    = 0.95
        CLIP_EPS  = 0.2
        VF_COEF   = 0.5
        ENT_COEF  = 0.01
        PPO_EPOCHS = 4

        all_states       = []
        all_actions      = []
        all_old_log_probs = []
        all_returns      = []
        all_advantages   = []

        for ep in episodes:
            # Only keep steps that received a reward assignment
            steps = [s for s in ep if 'reward' in s]
            if not steps:
                continue

            rewards = [s['reward'] for s in steps]
            values  = [float(s['value'].item() if hasattr(s['value'], 'item')
                             else s['value']) for s in steps]

            # GAE advantage + discounted return computation (backward pass)
            gae        = 0.0
            next_value = 0.0
            advantages = [0.0] * len(steps)
            returns    = [0.0] * len(steps)

            for t in reversed(range(len(steps))):
                delta          = rewards[t] + GAMMA * next_value - values[t]
                gae            = delta + GAMMA * LAMBDA * gae
                advantages[t]  = gae
                returns[t]     = gae + values[t]
                next_value     = values[t]

            for step, adv, ret in zip(steps, advantages, returns):
                all_states.append(step['state'])
                all_actions.append(step['action'])
                all_old_log_probs.append(step['log_prob'])
                all_advantages.append(adv)
                all_returns.append(ret)

        if not all_states:
            return

        # Stack into tensors
        states_t       = torch.cat(all_states, dim=0).to(self.device)
        actions_t      = torch.tensor(all_actions, dtype=torch.long).to(self.device)
        old_log_probs_t = torch.stack(all_old_log_probs).to(self.device).squeeze()
        advantages_t   = torch.tensor(all_advantages, dtype=torch.float32).to(self.device)
        returns_t      = torch.tensor(all_returns,    dtype=torch.float32).to(self.device)

        # Normalise advantages across the whole batch
        if advantages_t.numel() > 1 and advantages_t.std() > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = states_t.shape[0]

        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(n, device=self.device)

            # ── Policy loss ──────────────────────────────────────────
            logits       = self.policy_net(states_t[indices])
            dist         = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t[indices])
            entropy      = dist.entropy().mean()

            ratio  = torch.exp(new_log_probs - old_log_probs_t[indices].detach())
            adv_b  = advantages_t[indices]
            surr1  = ratio * adv_b
            surr2  = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean() - ENT_COEF * entropy

            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

            # ── Value loss ───────────────────────────────────────────
            values_pred = self.value_net(states_t[indices])
            value_loss  = VF_COEF * torch.nn.functional.mse_loss(
                values_pred, returns_t[indices].detach()
            )

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()