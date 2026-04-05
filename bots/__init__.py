"""
bots/__init__.py — Utility helpers used by runner scripts.

Provides:
  parse_players(spec_str)  -> list of (pid, btype, adapter)
  create_bot(btype)        -> BotAdapter
  escalate_blinds(...)     -> (sb, bb)
"""
import re
from core.engine import InProcessBot, RandomBot
from core.bot_api import BotAdapter, PlayerView, Action


# ── Bot creation ──────────────────────────────────────────────────────────────

def create_bot(btype: str) -> BotAdapter:
    """
    Create a bot adapter from a type string.

    Recognised types (case-insensitive):
      mc, mc<N>          MonteCarloBot (optional sim count: mc200, mc500)
      smart              SmartBot (heuristic)
      ml                 MLBot (supervised learning)
      rl, rl:<path>      RLBot (reinforcement learning, with optional model path)
      random             RandomBot
    """
    btype = btype.strip().lower()

    if btype.startswith("mc"):
        from bots.monte_carlo_bot import MonteCarloBot
        m = re.match(r"mc(\d+)", btype)
        sims = int(m.group(1)) if m else 200
        return _wrap(MonteCarloBot(simulations=sims))

    if btype in ("smart", "smartbot", "heuristic"):
        from bots.poker_mind_bot import SmartBot
        return _wrap(SmartBot())

    if btype in ("ml", "mlbot"):
        from bots.ml_bot import MLBot
        return _wrap(MLBot())

    if btype in ("rl", "rlbot") or btype.startswith("rl:"):
        from bots.rl_bot import RLBot
        if btype.startswith("rl:"):
            model_path = btype[3:]  # Extract path after "rl:"
            return _wrap(RLBot(model_path=model_path))
        else:
            return _wrap(RLBot())

    if btype in ("random",):
        return InProcessBot(RandomBot())

    raise ValueError(f"Unknown bot type: {btype!r}. "
                     "Expected one of: mc, mc<N>, smart, ml, rl, rl:<path>, random")


class _PlayerViewAdapter(BotAdapter):
    """Thin BotAdapter that passes PlayerView straight through."""
    def __init__(self, bot):
        self.bot = bot

    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)


def _wrap(bot) -> BotAdapter:
    """Wrap a bot object in a BotAdapter."""
    return _PlayerViewAdapter(bot)


# ── Player-spec parsing ────────────────────────────────────────────────────────

def parse_players(spec: str):
    """
    Parse a comma-separated player spec string into a list of
    (player_id, bot_type, adapter) tuples.

    Examples:
      "mc200,smart,ml,rl"
      "P1=mc200,P2=smart,P3=rl"

    Auto-assigns P1, P2, ... when no explicit IDs are given.
    """
    entries = [s.strip() for s in spec.split(",") if s.strip()]
    result = []
    for i, entry in enumerate(entries):
        if "=" in entry:
            pid, btype = entry.split("=", 1)
            pid, btype = pid.strip(), btype.strip()
        else:
            pid = f"P{i + 1}"
            btype = entry
        adapter = create_bot(btype)
        result.append((pid, btype, adapter))
    return result


# ── Blind escalation ──────────────────────────────────────────────────────────

def escalate_blinds(hand_count: int, base_sb: int, base_bb: int,
                    blind_increase_every: int) -> tuple:
    """
    Return (sb, bb) for the given hand number.

    Blinds increase 1.5x every `blind_increase_every` hands.
    If `blind_increase_every` is 0 (or negative), no escalation occurs.
    """
    if blind_increase_every <= 0:
        return base_sb, base_bb

    level = (hand_count - 1) // blind_increase_every
    multiplier = 1.5 ** level
    sb = max(1, int(base_sb * multiplier))
    bb = max(2, int(base_bb * multiplier))
    return sb, bb
