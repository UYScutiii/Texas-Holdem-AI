"""
Texas Hold'em Tournament UI
----------------------------
Run this file to open the tournament viewer.
Click "Play" to start the game and watch chip stacks update live.
"""

import sys
import time
import threading
import matplotlib
if sys.platform == "darwin":
    matplotlib.use("macosx")
else:
    import tkinter  # noqa: F401 – needed by TkAgg backend on Windows
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from collections import defaultdict

from core.engine import Table, Seat, InProcessBot, TournamentManager
from bots.monte_carlo_bot import MonteCarloBot
from bots.poker_mind_bot import PokerMindBot

# ─── TOURNAMENT SETTINGS ──────────────────────────────────────────────────────

PLAYERS = [
    ("Monte Carlo",  MonteCarloBot(simulations=200)),
    ("Poker Mind",   PokerMindBot()),
    ("Monte Carlo 2",MonteCarloBot(simulations=100)),
    ("Poker Mind 2", PokerMindBot()),
]

STARTING_CHIPS = 1000
SMALL_BLIND    = 5
BIG_BLIND      = 10
HAND_DELAY     = 0.05   # seconds between hands (lower = faster)

# ─── COLOURS ──────────────────────────────────────────────────────────────────

COLOURS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
           "#9C27B0", "#00BCD4", "#E91E63", "#8BC34A"]

# ──────────────────────────────────────────────────────────────────────────────


class TournamentUI:
    def __init__(self):
        self.player_ids   = [name for name, _ in PLAYERS]
        self.bots         = {name: InProcessBot(bot) for name, bot in PLAYERS}
        self.colours      = {pid: COLOURS[i % len(COLOURS)]
                             for i, pid in enumerate(self.player_ids)}

        self.chip_history: list[dict] = []
        self.running      = False
        self.finished     = False

        self._build_figure()

    # ── Figure setup ──────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 7), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("Texas Hold'em Tournament")

        # Main chart area
        self.ax = self.fig.add_axes([0.08, 0.18, 0.88, 0.72])
        self.ax.set_facecolor("#16213e")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#444")
        self.ax.set_xlabel("Hand", fontsize=12)
        self.ax.set_ylabel("Chips", fontsize=12)
        self.ax.set_title("Texas Hold'em Tournament", fontsize=15, fontweight="bold", pad=12)
        self.ax.grid(True, alpha=0.15, color="white")

        # Draw initial flat lines at starting chips
        self.lines = {}
        for pid in self.player_ids:
            line, = self.ax.plot(
                [0], [STARTING_CHIPS],
                label=pid,
                color=self.colours[pid],
                linewidth=2.5,
                alpha=0.9,
            )
            self.lines[pid] = line

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, STARTING_CHIPS * len(self.player_ids) * 1.05)
        self.ax.legend(loc="upper left", facecolor="#1a1a2e",
                       labelcolor="white", edgecolor="#444", fontsize=10)

        # Status label
        self.status_text = self.fig.text(
            0.5, 0.105, "Press  ▶ Play  to start the tournament",
            ha="center", va="center", fontsize=13,
            color="#aaaaaa", style="italic",
        )

        # Play button
        btn_ax = self.fig.add_axes([0.42, 0.02, 0.16, 0.07])
        self.play_btn = Button(
            btn_ax, "▶  Play",
            color="#0f3460", hovercolor="#e94560",
        )
        self.play_btn.label.set_color("white")
        self.play_btn.label.set_fontsize(13)
        self.play_btn.on_clicked(self._on_play)

    # ── Button handler ────────────────────────────────────────────────────────

    def _on_play(self, event=None):
        if self.running or self.finished:
            return
        self.running = True
        self._dirty = False
        self._winner_info = None
        self.play_btn.label.set_text("Running…")
        self.play_btn.color = "#333355"
        self.status_text.set_text("Tournament in progress…")
        self.fig.canvas.draw_idle()

        # Timer on main thread polls for data changes and redraws
        self._timer = self.fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._poll_redraw)
        self._timer.start()

        # Run tournament in background thread (data only, no GUI calls)
        t = threading.Thread(target=self._run_tournament, daemon=True)
        t.start()

    # ── Tournament loop ───────────────────────────────────────────────────────

    def _run_tournament(self):
        seats = [Seat(player_id=pid, chips=STARTING_CHIPS)
                 for pid in self.player_ids]
        by_pid = {s.player_id: s for s in seats}

        # Record hand 0
        snap = {"hand": 0, **{pid: STARTING_CHIPS for pid in self.player_ids}}
        self.chip_history.append(snap)
        self._mark_dirty()

        active_seats = list(seats)
        table        = Table()
        dealer       = 0
        hand_num     = 0
        finishing    = []          # (player_id, finish_position)
        total        = len(seats)

        while len(active_seats) > 1:
            hand_num += 1
            dealer_i = dealer % len(active_seats)

            try:
                table.play_hand(
                    active_seats, SMALL_BLIND, BIG_BLIND,
                    dealer_i, self.bots,
                )
            except Exception as e:
                print(f"[hand {hand_num}] error: {e}")
                break

            # Snapshot every player's chips (busted = 0)
            snap = {"hand": hand_num}
            for pid in self.player_ids:
                snap[pid] = by_pid[pid].chips
            self.chip_history.append(snap)
            self._mark_dirty()

            # Eliminate busted players
            eliminated = [s for s in active_seats if s.chips <= 0]
            for s in eliminated:
                pos = total - len(finishing)
                finishing.append((s.player_id, pos))
                active_seats.remove(s)
                print(f"  [OUT] {s.player_id} — position {pos}")

            dealer = (dealer + 1) % max(len(active_seats), 1)
            time.sleep(HAND_DELAY)

        # Winner
        if active_seats:
            finishing.append((active_seats[0].player_id, 1))
            winner = active_seats[0].player_id
        else:
            winner = "?"

        # Signal main thread to do final UI update
        self._signal_finish(winner, hand_num)

    # ── Data flag (called from background thread — no GUI calls) ────────────────

    def _mark_dirty(self):
        self._dirty = True

    def _signal_finish(self, winner: str, hands_played: int):
        self._winner_info = (winner, hands_played)
        self._dirty = True

    # ── Main-thread timer callback — safe to touch GUI here ──────────────────

    def _poll_redraw(self):
        if not self._dirty:
            return

        self._dirty = False

        # Redraw lines
        hands = [e["hand"] for e in self.chip_history]
        for pid, line in self.lines.items():
            y = [e.get(pid, 0) for e in self.chip_history]
            line.set_data(hands, y)
        self.ax.set_xlim(0, max(hands) + 1 if hands else 10)

        # Check if tournament ended
        if self._winner_info is not None:
            winner, hands_played = self._winner_info
            self._timer.stop()
            self.running = False
            self.finished = True
            colour = self.colours.get(winner, "white")
            self.status_text.set_text(
                f"🏆  {winner} wins!   ({hands_played} hands played)",
            )
            self.status_text.set_color(colour)
            self.play_btn.label.set_text("Done")
            print(f"\n🏆  Winner: {winner}  ({hands_played} hands)")

        self.fig.canvas.draw_idle()

    # ── Entry point ───────────────────────────────────────────────────────────

    def show(self):
        plt.show()


if __name__ == "__main__":
    ui = TournamentUI()
    ui.show()
