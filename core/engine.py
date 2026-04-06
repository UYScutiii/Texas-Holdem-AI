"""
Texas Hold'em Engine (Stable Fixed Version)
------------------------------------------
Now with:
- Correct indentation
- Safety breaker to avoid infinite loops
- Auto-reset of stacks if fewer than 2 players remain
"""
from __future__ import annotations
import random
import warnings
import pickle
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque, Counter
from itertools import combinations

from .bot_api import Action, PlayerView, BotAdapter
from core.logger import DecisionLogger # imports logger.py

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}
Card = Tuple[str, str]

# Maximum possible score from eval_hand / _score_five.
# Royal flush (AKQJT suited) = (8 << 24) | 12 = 134_217_740.
# Use this to normalise hand strength to [0, 1].
EVAL_HAND_MAX: int = (8 << 24) | 12  # 134_217_740

_FULL_DECK: List[Card] = [(r, s) for r in RANKS for s in SUITS]

def _build_five_card_table() -> dict:
    """
    Precompute scores for all C(52,5) = 2,598,960 five-card hands.
    Uses the original scoring logic exactly once at module load time.
    Returns a dict keyed by a canonical sorted tuple of (rank_int, suit) pairs.
    """
    def _score(cards):
        ranks = sorted([RANK_TO_INT[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        is_flush = len(set(suits)) == 1

        is_straight = False
        straight_high = 0
        if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
            is_straight = True
            straight_high = ranks[0]
        elif ranks == [12, 3, 2, 1, 0]:  # A-5-4-3-2 wheel
            is_straight = True
            straight_high = 3  # 5-high

        cnt = Counter(ranks)
        freq = sorted(cnt.values(), reverse=True)
        groups = sorted(cnt.keys(), key=lambda r: (cnt[r], r), reverse=True)

        def pack(g):
            v = 0
            for r in g:
                v = v * 15 + r
            return v

        if is_straight and is_flush:
            return (8 << 24) | straight_high
        if freq == [4, 1]:
            return (7 << 24) | pack(groups)
        if freq == [3, 2]:
            return (6 << 24) | pack(groups)
        if is_flush:
            return (5 << 24) | pack(ranks)
        if is_straight:
            return (4 << 24) | straight_high
        if freq[0] == 3:
            return (3 << 24) | pack(groups)
        if freq[:2] == [2, 2]:
            return (2 << 24) | pack(groups)
        if freq[0] == 2:
            return (1 << 24) | pack(groups)
        return (0 << 24) | pack(ranks)

    table = {}
    for combo in combinations(_FULL_DECK, 5):
        # Key: sorted by (rank_int desc, suit) for a canonical form.
        key = tuple(sorted(combo, key=lambda c: (-RANK_TO_INT[c[0]], c[1])))
        table[key] = _score(list(combo))
    return table


# Load from disk if cached; build and save on first run.
# Cache lives at models/five_card_table.pkl (≈ 50 MB).
_TABLE_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "five_card_table.pkl",
)

def _load_or_build_five_card_table() -> dict:
    if os.path.exists(_TABLE_CACHE):
        try:
            with open(_TABLE_CACHE, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass  # corrupted cache — fall through and rebuild
    print("[engine] Building 5-card lookup table (one-time, ~15s) …", flush=True)
    table = _build_five_card_table()
    os.makedirs(os.path.dirname(_TABLE_CACHE), exist_ok=True)
    with open(_TABLE_CACHE, "wb") as fh:
        pickle.dump(table, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[engine] Table cached to {_TABLE_CACHE}", flush=True)
    return table

_FIVE_CARD_TABLE: dict = _load_or_build_five_card_table()


def _score_five(cards: List[Card]) -> int:
    """
    Score exactly 5 cards via a precomputed lookup table.
    Higher = better.  Same output range as the original computation.
    """
    key = tuple(sorted(cards, key=lambda c: (-RANK_TO_INT[c[0]], c[1])))
    return _FIVE_CARD_TABLE[key]


def eval_hand(hole: List[Card], board: List[Card]) -> int:
    """
    Pure-Python 5-card hand evaluator. No external dependencies.
    Returns an integer where higher = better.

    Uses all combinations of hole + board cards and returns the best 5-card score.
    Falls back to a simple preflop heuristic when fewer than 3 board cards exist.
    """
    all_cards = list(hole) + list(board)
    if len(all_cards) >= 5:
        return max(_score_five(list(combo)) for combo in combinations(all_cards, 5))
    # Preflop / early street heuristic
    ranks = [c[0] for c in hole]
    base = sum(RANK_TO_INT.get(r, 0) for r in ranks)
    if len(ranks) == 2 and ranks[0] == ranks[1]:
        base += 40
    return base

def _compute_to_call(contrib, alive_pids, pid):
    """
    contrib: dict[player_id] -> contribution this street
    alive_pids: list of players still in the hand (not folded)
    pid: acting player

    Returns (to_call, highest_contrib).
    """
    highest = 0
    for p in alive_pids:
        c = contrib.get(p, 0)
        if c > highest:
            highest = c

    player_c = contrib.get(pid, 0)
    to_call = highest - player_c
    if to_call < 0:
        to_call = 0
    return to_call, highest

def _legal_actions_for(
    pid,
    seat,          # seat object for this player
    contrib,       # dict[player_id] -> contrib this street
    alive_pids,    # list of active players
    big_blind,     # numeric BB
):
    """
    Returns a list of action dicts like:
      {"type": "check"}
      {"type": "bet", "min": X, "max": Y}
      {"type": "call"}
      {"type": "raise", "min": X, "max": Y}
      {"type": "fold"}
    """
    legal = []
    to_call, highest = _compute_to_call(contrib, alive_pids, pid)
    chips = seat.chips

    # === CASE: player is NOT facing a bet (to_call == 0) ===
    if to_call == 0:
        # CASE A: no bet at all on this street yet (everyone at 0)
        everyone_zero = all(contrib.get(p, 0) == 0 for p in alive_pids)
        if everyone_zero:
            # Check or new bet
            legal.append({"type": "check"})
            if chips > 0:
                legal.append({"type": "bet", "min": big_blind, "max": chips})
        else:
            # CASE B: bet exists but this player has already matched it.
            # They may check, but NOT bet again at same level.
            legal.append({"type": "check"})

    # === CASE: facing a bet (to_call > 0) ===
    else:
        # fold is always legal
        legal.append({"type": "fold"})

        # call (possibly all-in)
        if chips <= to_call:
            # calling puts them all in
            legal.append({"type": "call"})  # your engine can interpret as all-in
        else:
            legal.append({"type": "call"})

            # raise only if they have more than to_call
            # New *total* contribution must be at least (highest + big_blind)
            min_total = highest + big_blind
            max_total = contrib.get(pid, 0) + chips  # everything they have

            if min_total > contrib.get(pid, 0) and max_total > min_total:
                legal.append({"type": "raise", "min": min_total, "max": max_total})

    return legal

def calculate_side_pots(contributions: Dict[str, int]) -> List[Dict]:
    """
    Split total per-player contributions into main pot + side pots.

    Args:
        contributions: {player_id: total_chips_put_in} across all streets.

    Returns:
        List of pots ordered from main to highest side pot.
        Each pot is {"amount": int, "eligible": list[str]}.
        A player is eligible for a pot only if they contributed at least
        up to that pot's threshold level.
    """
    contribs = {pid: amt for pid, amt in contributions.items() if amt > 0}
    if not contribs:
        return []

    levels = sorted(set(contribs.values()))
    pots = []
    prev_level = 0
    for level in levels:
        eligible = [pid for pid, amt in contribs.items() if amt >= level]
        pot_amount = (level - prev_level) * len(eligible)
        if pot_amount > 0:
            pots.append({"amount": pot_amount, "eligible": eligible})
        prev_level = level

    return pots

@dataclass
class Seat:
    player_id: str
    chips: int
    is_sitting_out: bool = False

class InProcessBot(BotAdapter):
    def __init__(self, bot_obj: Any):
        self.bot = bot_obj

    def act(self, view: PlayerView) -> Action:
        # Pass PlayerView directly; bots that still expect a dict get one via fallback
        try:
            a = self.bot.act(view)
        except (AttributeError, TypeError, Exception) as e:
            # Legacy bot expects a dict — convert for backwards compatibility
            warnings.warn(
                f"Bot for {view.me} failed with {type(e).__name__}: {e}, using dict fallback"
            )
            state = {
                "street": view.street,
                "position": view.position,
                "hole_cards": view.hole_cards,
                "board": view.board,
                "pot": view.pot,
                "to_call": view.to_call,
                "min_raise": view.min_raise,
                "max_raise": view.max_raise,
                "legal_actions": view.legal_actions,
                "stacks": view.stacks,
                "me": view.me,
                "opponents": view.opponents,
                "history": view.history,
            }
            a = self.bot.act(state)

        t = a.get("type") if isinstance(a, dict) else getattr(a, "type", None)
        amt = a.get("amount") if isinstance(a, dict) else getattr(a, "amount", None)
        return Action(t, amt)

class RandomBot:
    def act(self, state) -> Dict[str, Any]:
        if isinstance(state, PlayerView) or hasattr(state, "legal_actions"):
            legal = state.legal_actions
        else:
            legal = state["legal_actions"]
        choice = random.choice(legal)
        if choice["type"] in ("bet", "raise"):
            lo, hi = choice["min"], choice["max"]
            amt = random.randint(lo, hi)
            return {"type": choice["type"], "amount": amt}
        return {"type": choice["type"]}

class Table:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random(7331)
        self.hand_counter = 0

    def play_hand(self, seats: List[Seat | Dict[str, Any]], small_blind: int, big_blind: int,
                  dealer_index: int, bot_for: Dict[str, BotAdapter], on_event=None,
                  log_decisions: bool = False) -> Dict[str, int]:

        logger = DecisionLogger(enabled=log_decisions)

        # NEW — set hand ID
        logger.start_hand(self.hand_counter)
        self.hand_counter += 1

        # Normalize seats
        seats = [s if isinstance(s, Seat) else Seat(**s) for s in seats]
        by_pid = {s.player_id: s for s in seats}
        start_chips = {s.player_id: s.chips for s in seats}

        # Ensure at least 2 active players
        active = [s for s in seats if s.chips > 0 and not s.is_sitting_out]
        if len(active) < 2:
            raise ValueError("Not enough active players to play a hand")
        assert 2 <= len(active) <= 10

        # Determine play order (dealer rotation)
        order = deque(range(len(seats)))
        order.rotate(-dealer_index)
        ring = [i for i in order if seats[i].chips > 0 and not seats[i].is_sitting_out]

        # Assign positions
        positions = self._positions(len(ring))
        pos_by_pid = {seats[idx].player_id: pos for pos, idx in zip(positions, ring)}

        # Initialize per-player contributions (for current street)
        contrib = defaultdict(int, {s.player_id: 0 for s in seats if not s.is_sitting_out})

        def post_blind(kind: str, seat_index: int, amount: int):
            seat = seats[seat_index]
            amt = min(seat.chips, amount)
            seat.chips -= amt
            contrib[seat.player_id] += amt

        # Post blinds
        if len(ring) == 2:
            sb_idx, bb_idx = ring[0], ring[1]
        else:
            sb_idx, bb_idx = ring[1], ring[2 % len(ring)]
        post_blind("SB", sb_idx, small_blind)
        post_blind("BB", bb_idx, big_blind)

        # Shuffle and deal
        self._deck = list(_FULL_DECK)
        self.rng.shuffle(self._deck)
        hole = {seats[idx].player_id: [self._deck.pop(), self._deck.pop()] for idx in ring}
        board: List[Card] = []
        history: List[Any] = []

        # Preflop action starts at UTG (index 3 for 3+ players; index 0 for heads-up)
        # ring[0]=BTN, ring[1]=SB, ring[2]=BB, ring[3]=UTG (first to act preflop)
        preflop_start = 3 if len(ring) > 2 else 0

        streets = [
            ("preflop", self._betting_round, {"start_idx": preflop_start}),
            ("flop",    self._deal_flop_then_bet,  {}),
            ("turn",    self._deal_turn_then_bet,   {}),
            ("river",   self._deal_river_then_bet,  {}),
        ]

        # Total pot accumulated from completed streets
        pot_total = 0

        # Track each player's total contribution across all streets (for side pots)
        total_contrib = defaultdict(int)

        # --- main street loop ---
        for street_name, fn, extra_kwargs in streets:

            winner = fn(
                street_name,
                seats,
                ring,
                pos_by_pid,
                hole,
                board,
                contrib,
                pot_total,
                big_blind,
                bot_for,
                history,
                on_event,
                logger=logger,
                **extra_kwargs,
            )

            # If someone wins by everyone else folding
            if isinstance(winner, str):
                total_pot = pot_total + sum(contrib.values())
                by_pid[winner].chips += total_pot

                return {
                    pid: by_pid[pid].chips - start_chips.get(pid, by_pid[pid].chips)
                    for pid in start_chips
                }

            # No winner yet — accumulate this street's contribs
            for pid, c in contrib.items():
                total_contrib[pid] += c
            pot_total += sum(contrib.values())

            # Reset contrib for next street
            contrib = defaultdict(int, {seats[i].player_id: 0 for i in ring})

        # --- showdown ---
        # Accumulate final street's contributions
        for pid, c in contrib.items():
            total_contrib[pid] += c

        # Distribute pot using side pots
        share_net = self._showdown_and_settle(hole, board, total_contrib)

        # Apply showdown results
        for pid, delta in share_net.items():
            by_pid[pid].chips += delta

        # Final per-player net for this hand
        net = {
            pid: by_pid[pid].chips - start_chips.get(pid, by_pid[pid].chips)
            for pid in start_chips
        }

        # Log final results for ML training
        for pid, delta in net.items():
            logger.log_result(pid, delta)

        logger.flush()
        logger.close()

        return net

    def _deal_flop_then_bet(self, *a, **k):
        _, seats, ring, pos_by_pid, hole, board, contrib, pot, bb, bot_for, history, on_event = a
        board.extend([self._pop_card(), self._pop_card(), self._pop_card()])
        return self._betting_round(*a, **k)

    def _deal_turn_then_bet(self, *a, **k):
        _, seats, ring, pos_by_pid, hole, board, contrib, pot, bb, bot_for, history, on_event = a
        board.append(self._pop_card())
        return self._betting_round(*a, **k)

    def _deal_river_then_bet(self, *a, **k):
        _, seats, ring, pos_by_pid, hole, board, contrib, pot, bb, bot_for, history, on_event = a
        board.append(self._pop_card())
        return self._betting_round(*a, **k)

    def _pop_card(self):
        return self._deck.pop()

    def _betting_round(
        self, street, seats, ring, pos_by_pid, hole, board, contrib, pot, bb,
        bot_for, history, on_event, logger, start_idx: int = 0
    ):
        # print(f"\n=== BETTING ROUND START: {street} ===")
        # print(f"Pot before street: {pot}")
        # print("Ring order:", [seats[i].player_id for i in ring])
        # print("Initial contrib:", {s.player_id: contrib.get(s.player_id, 0) for s in seats})

        # Ensure contrib entries
        if not contrib:
            contrib = defaultdict(int, {s.player_id: 0 for s in seats if not s.is_sitting_out})
        else:
            for s in seats:
                contrib.setdefault(s.player_id, 0)

        folded = defaultdict(bool)
        allin = defaultdict(bool)

        current_bet = max(contrib.values()) if contrib else 0
        last_raise_size = bb if current_bet > 0 else bb

        # ---- helpers ----
        def num_players_can_act():
            cnt = 0
            for i in ring:
                s = seats[i]
                pid = s.player_id
                if folded[pid] or allin[pid] or s.chips <= 0:
                    continue
                cnt += 1
            return cnt

        def all_live_equal():
            live = []
            contribs = set()
            for i in ring:
                s = seats[i]
                pid = s.player_id
                if folded[pid]:
                    continue
                if allin[pid]:
                    continue
                if s.chips <= 0:
                    allin[pid] = True
                    continue
                live.append(pid)
                contribs.add(contrib[pid])
            if len(live) <= 1:
                return True
            return len(contribs) == 1

        idx = start_idx % len(ring)
        safety = 0

        # -------- MAIN LOOP ----------
        while True:
            safety += 1
            if safety > 500:
                # print("!!! SAFETY BREAK in betting_round")
                break

            if num_players_can_act() == 0:
                # print("No players able to act → ending round")
                break

            si = ring[idx]
            seat = seats[si]
            pid = seat.player_id

            # Skip dead players
            if folded[pid] or allin[pid] or seat.chips <= 0:
                idx = (idx + 1) % len(ring)
                if all_live_equal():
                    break
                continue

            # Recompute current bet / call amount
            current_bet = max(contrib.values()) if contrib else 0
            to_call = max(0, current_bet - contrib[pid])

            # LEGAL ACTIONS
            legal = []
            if to_call == 0:
                legal.append({"type": "check"})
                if seat.chips > 0:
                    min_bet = min(bb, seat.chips)
                    max_bet = seat.chips
                    if max_bet >= min_bet:
                        legal.append({"type": "bet", "min": min_bet, "max": max_bet})
            else:
                legal.append({"type": "fold"})
                call_amt = min(seat.chips, to_call)
                if call_amt > 0:
                    legal.append({"type": "call"})
                if seat.chips > to_call:
                    max_total = seat.chips + contrib[pid]
                    min_total = current_bet + last_raise_size
                    min_total = max(min_total, current_bet + bb)
                    if max_total >= min_total:
                        legal.append({
                            "type": "raise",
                            "min": min_total,
                            "max": max_total,
                        })

            # print(f"[{street}] Acting: {pid} | chips={seat.chips} contrib={contrib[pid]} to_call={to_call}")
            # print("    Legal:", legal)

            # PlayerView
            if to_call == 0:
                pv_min_raise = bb
                pv_max_raise = seat.chips
            else:
                pv_min_raise = max(0, (current_bet + last_raise_size) - contrib[pid])
                pv_max_raise = seat.chips

            view = PlayerView(
                me=pid,
                street=street,
                position=pos_by_pid[pid],
                hole_cards=hole[pid],
                board=list(board),
                pot=pot + sum(contrib.values()),
                to_call=to_call,
                min_raise=pv_min_raise,
                max_raise=pv_max_raise,
                legal_actions=legal,
                stacks={seats[i].player_id: seats[i].chips for i in ring},
                opponents=[seats[i].player_id for i in ring if seats[i].player_id != pid],
                history=list(history),
            )

            # BOT ACTION
            raw_action = bot_for[pid].act(view)

            # ---- ML LOGGING (only here, once per actual action) ----
            if logger is not None:
                logger.log_decision({
                    "player": pid,
                    "street": street,
                    "hole": hole[pid],
                    "board": list(board),
                    "pot": view.pot,
                    "to_call": to_call,
                    "legal": view.legal_actions,
                    "chosen_action": {"type": raw_action.type, "amount": raw_action.amount},
                    "stacks": view.stacks,
                    "opponents": view.opponents,
                    "folded": False,
                    "hand_id": getattr(self, "hand_index", None)
                })

            # Sanitize illegal action type
            legal_types = {a["type"] for a in legal}
            action_type = raw_action.type
            action_amt = raw_action.amount

            if action_type not in legal_types:
                print(f"    [WARN] Illegal action '{action_type}', fixing...")
                if "call" in legal_types:
                    action_type = "call"; action_amt = None
                elif "check" in legal_types:
                    action_type = "check"; action_amt = None
                else:
                    action_type = "fold"; action_amt = None

            # Sanitize bet/raise amount
            if action_type in ("bet", "raise"):
                spec = next(a for a in legal if a["type"] == action_type)
                lo, hi = spec["min"], spec["max"]
                if action_amt is None:
                    action_amt = lo
                amt = int(action_amt)
                if amt < lo: amt = lo
                if amt > hi: amt = hi
                action_amt = amt
            else:
                action_amt = None

            action = Action(action_type, action_amt)
            # print(f"    Chosen action: {action.type} {action.amount}")

            # Add to history BEFORE modifying contrib
            history.append({
                "street": street,
                "pid": pid,
                "type": action.type,
                "amount": action.amount,
                "to_call_before": to_call,
            })

            # APPLY ACTION
            if action.type == "fold":
                folded[pid] = True
                hole[pid] = []
                # print(f"    {pid} FOLDS")

            elif action.type == "call":
                need = min(seat.chips, to_call)
                seat.chips -= need
                contrib[pid] += need
                if seat.chips <= 0:
                    allin[pid] = True
                # print(f"    {pid} CALLS {need}")

            elif action.type == "check":
                pass  # nothing to do for a check

            elif action.type in ("bet", "raise"):
                prev_bet = max(contrib.values())
                target_total = int(action.amount or 0)
                need = max(0, target_total - contrib[pid])
                if need > seat.chips:
                    need = seat.chips
                    target_total = contrib[pid] + need

                seat.chips -= need
                contrib[pid] += need
                if seat.chips <= 0:
                    allin[pid] = True

                new_bet = max(contrib.values())
                if street == "preflop" and prev_bet == 0:
                    last_raise_size = new_bet
                else:
                    raise_sz = new_bet - prev_bet
                    if raise_sz > 0:
                        last_raise_size = raise_sz

                # print(f"    {pid} {action.type.upper()} to {target_total} (paid {need})")

            if all_live_equal():
                # print("All live equal → ending street")
                break
            if num_players_can_act() == 0:
                # print("No one left who can act → ending street")
                break

            idx = (idx + 1) % len(ring)

        alive = [seats[i].player_id for i in ring if not folded[seats[i].player_id]]
        # print(f"=== BETTING ROUND END: {street} | Alive: {alive} ===")

        if len(alive) == 1:
            # print(f"--> Winner by fold on {street}: {alive[0]}")
            return alive[0]
        return None


    def _showdown_and_settle(self, hole, board, total_contrib):
        """Distribute winnings using side pots so all-in players only
        compete for the portion of the pot they contributed to."""
        net = {pid: 0 for pid in hole}

        if not total_contrib or sum(total_contrib.values()) <= 0:
            return net

        # Players still in the hand (not folded)
        eligible = {pid: cards for pid, cards in hole.items() if cards and len(cards) == 2}
        if not eligible:
            return net

        ranks = {pid: eval_hand(cards, board) for pid, cards in eligible.items()}
        pots = calculate_side_pots(total_contrib)

        for pot in pots:
            # Intersect pot-eligible (contributed enough) with showdown-eligible (didn't fold)
            contenders = [pid for pid in pot["eligible"] if pid in ranks]
            if not contenders:
                # Everyone who contributed to this pot folded — give to any remaining player
                contenders = [pid for pid in eligible]
            if not contenders:
                continue

            best = max(ranks[pid] for pid in contenders)
            winners = [pid for pid in contenders if ranks[pid] == best]
            n_winners = len(winners)
            base_share = pot["amount"] // n_winners
            remainder = pot["amount"] % n_winners
            for i, w in enumerate(winners):
                net[w] += base_share + (1 if i < remainder else 0)

        return net



    def _positions(self, n):
        if n == 2:
            return ["BTN", "BB"]
        tags = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"]
        return tags[:n]

class TournamentManager:
    def __init__(self, table: Table):
        self.table = table

    def run(self, seats, bot_for, small_blind, big_blind, dealer_index=0,
            on_event=None, live_graph=True):
        seats = [s if isinstance(s, Seat) else Seat(**s) for s in seats]
        active_seats = list(seats)
        dealer = dealer_index
        hand_number = 0
        chip_history: List[Dict] = []
        # finishing_order: list of (player_id, position) where position 1 = winner
        finishing_order: List[Tuple[str, int]] = []
        total_players = len(seats)

        # Set up live graph
        player_ids = [s.player_id for s in seats]
        graph = LiveTournamentGraph(player_ids) if live_graph else None

        # Record initial chip snapshot
        snapshot = {"hand": 0}
        for s in seats:
            snapshot[s.player_id] = s.chips
        chip_history.append(snapshot)

        if graph:
            graph.update(chip_history)

        while len(active_seats) > 1:
            hand_number += 1
            # Fix dealer index if it's out of range
            dealer = dealer % len(active_seats)

            self.table.play_hand(
                active_seats, small_blind, big_blind, dealer, bot_for, on_event=on_event
            )

            # Record chip snapshot after this hand
            snapshot = {"hand": hand_number}
            for s in seats:
                snapshot[s.player_id] = s.chips
            chip_history.append(snapshot)

            # Update live graph
            if graph:
                graph.update(chip_history)

            # Eliminate busted players (chips <= 0)
            eliminated = [s for s in active_seats if s.chips <= 0]
            for s in eliminated:
                # Last place = total_players, first eliminated gets worst position
                position = total_players - len(finishing_order)
                finishing_order.append((s.player_id, position))
                active_seats.remove(s)
                print(f"  [ELIMINATED] {s.player_id} finishes in position {position}")

            # Advance dealer
            if active_seats:
                dealer = (dealer + 1) % len(active_seats)

        # The last player standing is the winner (position 1)
        if active_seats:
            finishing_order.append((active_seats[0].player_id, 1))
            print(f"  [WINNER] {active_seats[0].player_id} wins the tournament!")

        # Finalize graph — save and keep window open
        if graph:
            graph.finish()

        # Build results dict: player_id -> finishing position
        results = {pid: pos for pid, pos in finishing_order}

        return {
            "results": results,
            "chip_history": chip_history,
            "hands_played": hand_number,
            "final_stacks": {s.player_id: s.chips for s in seats},
        }


class LiveTournamentGraph:
    """Real-time matplotlib graph that updates after every hand."""

    def __init__(self, player_ids: List[str]):
        import matplotlib
        for backend in ("macosx", "TkAgg", "Agg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue
        import matplotlib.pyplot as plt
        self._plt = plt

        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(12, 6))
        self._lines = {}
        for pid in player_ids:
            line, = self._ax.plot([], [], label=pid, linewidth=2)
            self._lines[pid] = line

        self._ax.set_title("Tournament Chip Stacks")
        self._ax.set_xlabel("Hand")
        self._ax.set_ylabel("Chips")
        self._ax.legend(loc="upper left")
        self._ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        self._hands: List[int] = []
        self._stacks: Dict[str, List[int]] = {pid: [] for pid in player_ids}

    def update(self, chip_history: List[Dict]):
        """Redraw all lines from the full chip_history."""
        self._hands = [entry["hand"] for entry in chip_history]

        for pid, line in self._lines.items():
            y = [entry.get(pid, 0) for entry in chip_history]
            self._stacks[pid] = y
            line.set_data(self._hands, y)

        self._ax.set_xlim(0, max(self._hands) if self._hands else 1)
        all_chips = [c for vals in self._stacks.values() for c in vals]
        self._ax.set_ylim(0, max(all_chips) * 1.1 if all_chips else 1)
        self._ax.legend(loc="upper left")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def finish(self, filename: str = "tournament_results.png"):
        """Save final chart and keep window open for viewing."""
        self._plt.ioff()
        self._fig.savefig(filename)
        print(f"Tournament chart saved to {filename}")
        self._plt.show()

