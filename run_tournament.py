"""
Texas Hold'em Tournament UI
----------------------------
Run this file to open the tournament viewer.
Click "Play" to start the game and watch chip stacks update live.

Speed controls:
  + / =   Speed up (less delay between hands)
  - / _   Slow down (more delay)
  Space   Pause / resume
"""

import sys
import time
import argparse
import threading
import matplotlib

# Auto-detect a working interactive backend
if sys.platform == "darwin":
    try:
        matplotlib.use("macosx")
    except Exception:
        matplotlib.use("TkAgg")
else:
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg")
    except ImportError:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

from core.engine import Table, Seat
from bots import parse_players, escalate_blinds

# ─── DEFAULTS ────────────────────────────────────────────────────────────────

DEFAULT_PLAYERS = "mc200,smart,ml,rl,cfr,icm,exploitative,gto,opponentmodel"
DEFAULT_CHIPS   = 1000
DEFAULT_SMALLBLIND = 5
DEFAULT_BIGBLIND   = 10
DEFAULT_DELAY   = 0.05
DEFAULT_BLIND_INCREASE_EVERY = 50

COLOURS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
           "#9C27B0", "#00BCD4", "#E91E63", "#8BC34A"]

SPEED_STEPS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]

# ─────────────────────────────────────────────────────────────────────────────


class TournamentUI:
    def __init__(self, players, starting_chips, base_sb, base_bb,
                 hand_delay, blind_increase_every):
        self.player_specs = players  # [(pid, btype, adapter), ...]
        self.player_ids   = [pid for pid, _, _ in players]
        self.bot_types    = {pid: btype for pid, btype, _ in players}
        self.bots         = {pid: adapter for pid, _, adapter in players}
        self.colours      = {pid: COLOURS[i % len(COLOURS)]
                             for i, pid in enumerate(self.player_ids)}

        self.starting_chips = starting_chips
        self.base_sb = base_sb
        self.base_bb = base_bb
        self.blind_increase_every = blind_increase_every

        self.hand_delay   = hand_delay
        self._speed_idx   = SPEED_STEPS.index(hand_delay) if hand_delay in SPEED_STEPS else 3
        self._paused      = False

        self.chip_history: list[dict] = []
        self.running      = False
        self.finished     = False
        self._current_blinds = (base_sb, base_bb)
        self._cancel_event   = threading.Event()
        self._cancelled      = False
        self._btn_mode       = "play"  # "play" | "cancel" | "restart"
        self._eliminations       = {}   # pid -> finishing position
        self._elimination_events = []   # [(pid, hand_num, finishing_pos), ...]
        self._elim_artists       = []   # matplotlib artists to remove each cycle
        self._summary_ax         = None  # overlay axes shown at tournament end
        self._highlights         = []   # list of {hand, text, color, age} dicts
        self._last_chip_leader   = None  # tracks chip leader for change detection
        self._last_ylim          = starting_chips  # last y-axis ceiling set on chart

        self._build_figure()

    # ── Figure setup ──────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("Texas Hold'em Tournament")

        # Main chart area — centered between highlights sidebar (left) and leaderboard (right)
        self.ax = self.fig.add_axes([0.22, 0.24, 0.44, 0.66])
        self.ax.set_facecolor("#16213e")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#444")
        self.ax.set_xlabel("Hand", fontsize=12)
        self.ax.set_ylabel("Chips", fontsize=12)
        self.ax.set_title("Texas Hold'em Tournament", fontsize=15,
                          fontweight="bold", pad=12)
        self.ax.grid(True, alpha=0.15, color="white")

        # Draw initial flat lines at starting chips
        self.lines = {}
        for pid in self.player_ids:
            label = f"{pid} ({self.bot_types[pid]})"
            line, = self.ax.plot(
                [0], [self.starting_chips],
                label=label,
                color=self.colours[pid],
                linewidth=2.5,
                alpha=0.9,
            )
            self.lines[pid] = line

        total_chips = self.starting_chips * len(self.player_ids)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, total_chips * 1.05)
        self.ax.legend(loc="upper left", facecolor="#1a1a2e",
                       labelcolor="white", edgecolor="#444", fontsize=9)

        # Leaderboard sidebar (right)
        self.lb_ax = self.fig.add_axes([0.69, 0.24, 0.29, 0.66])
        self.lb_ax.set_facecolor("#16213e")
        self.lb_ax.set_xticks([])
        self.lb_ax.set_yticks([])
        for spine in self.lb_ax.spines.values():
            spine.set_visible(False)
        self._draw_leaderboard()

        # Highlights feed sidebar (left)
        self.feed_ax = self.fig.add_axes([0.02, 0.24, 0.18, 0.66])
        self.feed_ax.set_facecolor("#16213e")
        self.feed_ax.set_xticks([])
        self.feed_ax.set_yticks([])
        for spine in self.feed_ax.spines.values():
            spine.set_visible(False)
        self._draw_feed()

        # Status label
        self.status_text = self.fig.text(
            0.5, 0.105, "Press  Play  to start  |  +/- speed  |  Space pause",
            ha="center", va="center", fontsize=13,
            color="#aaaaaa", style="italic",
        )

        # Blinds label (top-right of chart area)
        self.blinds_text = self.ax.text(
            0.98, 0.97,
            f"Blinds: {self.base_sb}/{self.base_bb}",
            transform=self.ax.transAxes,
            ha="right", va="top", fontsize=11,
            color="#ffcc00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor="#ffcc00", alpha=0.8),
        )

        # Play button
        btn_ax = self.fig.add_axes([0.42, 0.02, 0.16, 0.07])
        self.play_btn = Button(
            btn_ax, "Play",
            color="#0f3460", hovercolor="#e94560",
        )
        self.play_btn.label.set_color("white")
        self.play_btn.label.set_fontsize(13)
        self.play_btn.on_clicked(self._on_button_click)

        # Keyboard handler
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Keyboard / speed control ──────────────────────────────────────────────

    def _on_key(self, event):
        if event.key in ("+", "="):
            self._speed_idx = max(0, self._speed_idx - 1)
            self.hand_delay = SPEED_STEPS[self._speed_idx]
            self._update_speed_label()
        elif event.key in ("-", "_"):
            self._speed_idx = min(len(SPEED_STEPS) - 1, self._speed_idx + 1)
            self.hand_delay = SPEED_STEPS[self._speed_idx]
            self._update_speed_label()
        elif event.key == " ":
            self._paused = not self._paused
            self._update_speed_label()

    def _update_speed_label(self):
        if self._paused:
            state = "PAUSED"
        else:
            state = f"delay={self.hand_delay:.2f}s"
        self.status_text.set_text(f"Running  |  {state}  |  +/- speed  |  Space pause")
        self.fig.canvas.draw_idle()

    # ── Button dispatcher ─────────────────────────────────────────────────────

    def _on_button_click(self, event=None):
        if self._btn_mode == "play":
            self._start_tournament()
        elif self._btn_mode == "cancel":
            self._request_cancel()
        elif self._btn_mode == "restart":
            self._reset_to_play()

    def _start_tournament(self):
        if self.running or self.finished:
            return
        self._cancel_event.clear()
        self._cancelled = False
        self.running = True
        self._dirty = False
        self._winner_info = None
        # Switch to Cancel mode
        self._btn_mode = "cancel"
        self.play_btn.label.set_text("Cancel")
        self.play_btn.color = "#7a1a1a"
        self.play_btn.hovercolor = "#c0392b"
        self.play_btn.ax.set_facecolor("#7a1a1a")
        self.status_text.set_text("Tournament in progress...  |  +/- speed  |  Space pause")
        self.status_text.set_color("#aaaaaa")
        self.status_text.set_style("italic")
        self.fig.canvas.draw_idle()

        self._timer = self.fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._poll_redraw)
        self._timer.start()

        t = threading.Thread(target=self._run_tournament, daemon=True)
        t.start()

    def _request_cancel(self):
        """Signal the tournament thread to stop; UI reset happens in _poll_redraw."""
        self._cancel_event.set()
        self.play_btn.label.set_text("Stopping...")
        self.play_btn.color = "#444444"
        self.play_btn.ax.set_facecolor("#444444")
        self.status_text.set_text("Cancelling...")
        self.fig.canvas.draw_idle()

    def _reset_to_play(self):
        """Reset all state and restore the chart to the initial ready state."""
        # Clear data state first so leaderboard reads clean values
        self.chip_history        = []
        self._eliminations       = {}
        self._elimination_events = []
        # Remove any leftover marker artists from the main chart
        for artist in self._elim_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._elim_artists = []
        # Remove summary overlay if present
        if self._summary_ax is not None:
            try:
                self._summary_ax.remove()
            except Exception:
                pass
            self._summary_ax = None
        # Clear highlights feed
        self._highlights = []
        self._last_chip_leader = None
        self.running       = False
        self.finished      = False
        self._cancelled    = False
        self._cancel_event.clear()
        self._winner_info  = None
        self._current_blinds = (self.base_sb, self.base_bb)
        # Restore button to Play
        self._btn_mode = "play"
        self.play_btn.label.set_text("Play")
        self.play_btn.color = "#0f3460"
        self.play_btn.hovercolor = "#e94560"
        self.play_btn.ax.set_facecolor("#0f3460")
        # Restore status text
        self.status_text.set_text(
            "Press  Play  to start  |  +/- speed  |  Space pause")
        self.status_text.set_color("#aaaaaa")
        self.status_text.set_style("italic")
        # Restore chart lines to flat starting chips
        for pid, line in self.lines.items():
            line.set_data([0], [self.starting_chips])
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, self.starting_chips)
        self._last_ylim = self.starting_chips
        self.blinds_text.set_text(
            f"Blinds: {self.base_sb}/{self.base_bb}")
        # Explicitly wipe the leaderboard axes so no stale artists remain,
        # then re-apply sidebar styling before the fresh draw
        self.lb_ax.clear()
        self.lb_ax.set_facecolor("#16213e")
        self.lb_ax.set_xticks([])
        self.lb_ax.set_yticks([])
        for spine in self.lb_ax.spines.values():
            spine.set_visible(False)
        self._draw_leaderboard()
        self._draw_feed()
        self.fig.canvas.draw_idle()

    # ── Tournament loop ───────────────────────────────────────────────────────

    def _run_tournament(self):
        seats = [Seat(player_id=pid, chips=self.starting_chips)
                 for pid in self.player_ids]
        by_pid = {s.player_id: s for s in seats}

        snap = {"hand": 0, **{pid: self.starting_chips for pid in self.player_ids}}
        self.chip_history.append(snap)
        self._mark_dirty()

        active_seats = list(seats)
        table        = Table()
        dealer       = 0
        hand_num     = 0
        finishing     = []
        total        = len(seats)

        while len(active_seats) > 1:
            # Cancellation check (runs before every hand)
            if self._cancel_event.is_set():
                self._signal_cancelled()
                return
            # Pause loop — also exits immediately on cancel
            while self._paused and not self._cancel_event.is_set():
                time.sleep(0.05)

            hand_num += 1
            sb, bb = escalate_blinds(hand_num, self.base_sb, self.base_bb,
                                     self.blind_increase_every)
            self._current_blinds = (sb, bb)
            dealer_i = dealer % len(active_seats)

            active_bots = {s.player_id: self.bots[s.player_id]
                           for s in active_seats}

            try:
                table.play_hand(
                    active_seats, sb, bb,
                    dealer_i, active_bots,
                )
            except Exception as e:
                print(f"[hand {hand_num}] error: {e}")
                break

            snap = {"hand": hand_num}
            for pid in self.player_ids:
                snap[pid] = by_pid[pid].chips
            self.chip_history.append(snap)
            self._detect_highlights(hand_num)
            self._mark_dirty()

            eliminated = [s for s in active_seats if s.chips <= 0]
            for s in eliminated:
                pos = total - len(finishing)
                finishing.append((s.player_id, pos))
                self._eliminations[s.player_id] = pos
                self._elimination_events.append((s.player_id, hand_num, pos))
                active_seats.remove(s)
                print(f"  [OUT] {s.player_id} — position {pos}")

            dealer = (dealer + 1) % max(len(active_seats), 1)
            time.sleep(self.hand_delay)

        if active_seats:
            finishing.append((active_seats[0].player_id, 1))
            winner = active_seats[0].player_id
        else:
            winner = "?"

        self._signal_finish(winner, hand_num)

    # ── Leaderboard ───────────────────────────────────────────────────────────

    _ORDINALS = ["1st", "2nd", "3rd", "4th", "5th",
                 "6th", "7th", "8th", "9th", "10th"]

    def _ordinal(self, n: int) -> str:
        if 1 <= n <= len(self._ORDINALS):
            return self._ORDINALS[n - 1]
        return f"{n}th"

    def _draw_leaderboard(self):
        ax = self.lb_ax
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Title
        ax.text(0.5, 0.97, "Leaderboard",
                ha="center", va="top",
                fontsize=12, fontweight="bold", color="white",
                transform=ax.transAxes)

        # Separator line under title
        ax.axhline(y=0.93, xmin=0.05, xmax=0.95,
                   color="#444", linewidth=0.8)

        # Build current chip snapshot
        if self.chip_history:
            snap = self.chip_history[-1]
        else:
            snap = {pid: self.starting_chips for pid in self.player_ids}

        total_chips = self.starting_chips * len(self.player_ids)

        # Sort: active players descending by chips, then eliminated by position asc
        active  = [(pid, snap.get(pid, 0))
                   for pid in self.player_ids
                   if pid not in self._eliminations]
        active.sort(key=lambda x: x[1], reverse=True)

        eliminated = [(pid, self._eliminations[pid])
                      for pid in self.player_ids
                      if pid in self._eliminations]
        eliminated.sort(key=lambda x: x[1])  # best finish first

        rows = [(pid, snap.get(pid, 0), False) for pid, _ in active] + \
               [(pid, 0,                True)  for pid, _ in eliminated]

        n = len(rows)
        # Vertical layout: spread rows between y=0.90 and y=0.02
        row_h = 0.88 / max(n, 1)

        for rank, (pid, chips, is_out) in enumerate(rows, start=1):
            y_center = 0.90 - (rank - 0.5) * row_h
            color    = self._eliminations_color(pid, is_out)
            btype    = self.bot_types[pid]

            # Rank label
            ax.text(0.03, y_center, self._ordinal(rank),
                    ha="left", va="center",
                    fontsize=8, color="#888888",
                    transform=ax.transAxes)

            # Colored dot
            dot_color = color if not is_out else "#555555"
            ax.plot(0.18, y_center, "s",
                    color=dot_color, markersize=7,
                    transform=ax.transAxes, clip_on=False)

            # Player name + bot type
            name_color = color if not is_out else "#555555"
            ax.text(0.25, y_center, f"{pid} ({btype})",
                    ha="left", va="center",
                    fontsize=8, color=name_color,
                    transform=ax.transAxes)

            if is_out:
                # Finishing position
                elim_pos = self._eliminations[pid]
                ax.text(0.97, y_center,
                        f"OUT ({self._ordinal(elim_pos)})",
                        ha="right", va="center",
                        fontsize=7.5, color="#555555",
                        transform=ax.transAxes)
            else:
                # Chip count
                ax.text(0.97, y_center + row_h * 0.18,
                        f"{chips:,}",
                        ha="right", va="center",
                        fontsize=8, color="white",
                        transform=ax.transAxes)
                # Progress bar
                bar_w = max(chips / total_chips, 0.0) * 0.72
                bar_y = y_center - row_h * 0.25
                bar_h = row_h * 0.18
                # Background track
                ax.add_patch(mpatches.FancyBboxPatch(
                    (0.25, bar_y), 0.72, bar_h,
                    boxstyle="round,pad=0",
                    facecolor="#2a2a4a", edgecolor="none",
                    transform=ax.transAxes, clip_on=False))
                # Fill
                if bar_w > 0:
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (0.25, bar_y), bar_w, bar_h,
                        boxstyle="round,pad=0",
                        facecolor=self.colours[pid],
                        edgecolor="none", alpha=0.75,
                        transform=ax.transAxes, clip_on=False))

    def _eliminations_color(self, pid: str, is_out: bool) -> str:
        return self.colours.get(pid, "#ffffff")

    # ── Data flag ─────────────────────────────────────────────────────────────

    def _mark_dirty(self):
        self._dirty = True

    def _signal_cancelled(self):
        """Called from the tournament thread when the cancel event fires."""
        self._cancelled = True
        self._dirty = True

    def _signal_finish(self, winner: str, hands_played: int):
        self._winner_info = (winner, hands_played)
        self._dirty = True

    # ── Main-thread timer callback ────────────────────────────────────────────

    def _poll_redraw(self):
        if not self._dirty:
            return
        self._dirty = False

        hands = [e["hand"] for e in self.chip_history]
        for pid, line in self.lines.items():
            y = [e.get(pid, 0) for e in self.chip_history]
            line.set_data(hands, y)
        self.ax.set_xlim(0, max(hands) + 1 if hands else 10)

        # ── Dynamic y-axis scaling ────────────────────────────────────────────
        if self.chip_history:
            last_snap = self.chip_history[-1]
            active_chips = [
                last_snap.get(pid, 0)
                for pid in self.player_ids
                if last_snap.get(pid, 0) > 0
            ]
            if active_chips:
                max_chips = max(active_chips)
                y_max = max(self.starting_chips, max_chips * 1.15)
                if abs(y_max - self._last_ylim) / self._last_ylim > 0.05:
                    self.ax.set_ylim(0, y_max)
                    self._last_ylim = y_max

        # Update blinds display
        sb, bb = self._current_blinds
        self.blinds_text.set_text(f"Blinds: {sb}/{bb}")

        # ── Elimination markers ───────────────────────────────────────────────
        # Remove previous cycle's artists
        for artist in self._elim_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._elim_artists = []

        if self._elimination_events:
            total_chips = self.starting_chips * len(self.player_ids)
            # Sort by finishing position descending (worst finisher first / lowest y)
            events_sorted = sorted(self._elimination_events,
                                   key=lambda e: e[2], reverse=True)
            # Assign staggered y-offsets for events clustered within 5 hands
            offsets = {}  # index -> y_offset
            base_offsets = [total_chips * f
                            for f in (0.03, 0.08, 0.13, 0.18, 0.23, 0.28)]
            placed = []  # list of (hand_num, offset_level) already placed
            for ev in events_sorted:
                pid, hand, pos = ev
                colour = self.colours.get(pid, "white")
                # X marker at y=0
                pts = self.ax.plot(
                    hand, 0, "X",
                    color=colour, markersize=12,
                    markeredgewidth=2.5, zorder=10,
                )
                self._elim_artists.extend(pts)
                # Determine stagger level: count how many already-placed markers
                # are within 5 hands
                nearby_levels = [lvl for (h, lvl) in placed
                                 if abs(h - hand) <= 5]
                level = 0
                while level in nearby_levels:
                    level += 1
                placed.append((hand, level))
                y_off = base_offsets[min(level, len(base_offsets) - 1)]
                ordinal = self._ordinal(pos)
                ann = self.ax.annotate(
                    f"{pid} ({ordinal})",
                    xy=(hand, 0),
                    xytext=(hand, y_off),
                    fontsize=7,
                    color=colour,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="#1a1a2e",
                              edgecolor="none",
                              alpha=0.7),
                    arrowprops=dict(arrowstyle="-",
                                   color=colour,
                                   alpha=0.4,
                                   lw=0.8),
                    zorder=9,
                )
                self._elim_artists.append(ann)

        # Cancellation: reset everything back to ready state
        if self._cancelled:
            self._timer.stop()
            self.running = False
            self._reset_to_play()
            return

        if self._winner_info is not None:
            winner, hands_played = self._winner_info
            self._timer.stop()
            self.running = False
            self.finished = True
            colour = self.colours.get(winner, "white")
            self.status_text.set_text(
                f"{winner} wins!   ({hands_played} hands played)")
            self.status_text.set_color(colour)
            self.status_text.set_style("normal")
            # Switch to Restart mode
            self._btn_mode = "restart"
            self.play_btn.label.set_text("Restart")
            self.play_btn.color = "#0f3460"
            self.play_btn.hovercolor = "#e94560"
            self.play_btn.ax.set_facecolor("#0f3460")
            print(f"\nWinner: {winner}  ({hands_played} hands)")
            self._show_summary_overlay(winner, hands_played)

        self._draw_leaderboard()
        self._draw_feed()
        self.fig.canvas.draw_idle()

    # ── Summary overlay ───────────────────────────────────────────────────────

    def _show_summary_overlay(self, winner: str, hands_played: int):
        """Draw a semi-transparent end-of-tournament summary panel."""
        # Only show once; guard against repeated _poll_redraw calls
        if self._summary_ax is not None:
            return

        ax = self.fig.add_axes([0.24, 0.27, 0.40, 0.56])
        ax.set_facecolor("#1a1a2e")
        ax.patch.set_alpha(0.92)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#ffcc00")
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self._summary_ax = ax

        # ── Title ─────────────────────────────────────────────────────────────
        ax.text(0.5, 0.93, "Tournament Complete",
                ha="center", va="top",
                fontsize=14, fontweight="bold", color="white",
                transform=ax.transAxes)

        # ── Winner line ───────────────────────────────────────────────────────
        winner_color = self.colours.get(winner, "#ffcc00")
        btype = self.bot_types.get(winner, "?")
        ax.text(0.5, 0.82,
                f"Winner: ",
                ha="right", va="top",
                fontsize=12, color="#ffcc00",
                transform=ax.transAxes)
        ax.text(0.5, 0.82,
                f"{winner} ({btype})",
                ha="left", va="top",
                fontsize=12, color=winner_color,
                transform=ax.transAxes)

        # ── Hands played ──────────────────────────────────────────────────────
        ax.text(0.5, 0.73,
                f"Hands played: {hands_played}",
                ha="center", va="top",
                fontsize=10, color="white",
                transform=ax.transAxes)

        # ── Separator ─────────────────────────────────────────────────────────
        ax.axhline(y=0.66, xmin=0.04, xmax=0.96,
                   color="#ffcc00", linewidth=0.8, alpha=0.5)

        # ── Results table ─────────────────────────────────────────────────────
        # Build finishing order: 1st = winner, then eliminated sorted best→worst
        # _elimination_events: [(pid, hand_num, finishing_pos), ...]
        # finishing_pos: 1 = best (winner already not in _eliminations)
        elim_by_pid = {pid: (hand, pos)
                       for pid, hand, pos in self._elimination_events}

        # Sort all players by finishing position ascending (1st first)
        all_players = list(self.player_ids)
        def _finish_pos(pid):
            if pid == winner:
                return 1
            _, pos = elim_by_pid.get(pid, (0, len(all_players)))
            return pos

        ordered = sorted(all_players, key=_finish_pos)

        row_step = 0.065
        y_start  = 0.61  # just below separator

        for i, pid in enumerate(ordered):
            y = y_start - i * row_step
            if y < 0.02:  # don't draw off the bottom
                break

            pos      = _finish_pos(pid)
            ordinal  = self._ordinal(pos)
            btype_r  = self.bot_types.get(pid, "?")
            is_winner = (pid == winner)

            if pid in elim_by_pid:
                hand_survived, _ = elim_by_pid[pid]
            else:
                hand_survived = hands_played  # winner

            # Finishing position label
            ax.text(0.04, y, ordinal,
                    ha="left", va="top",
                    fontsize=9, color="#888888",
                    transform=ax.transAxes)

            # Player name + bot type
            name_color = (self.colours.get(pid, "white")
                          if is_winner
                          else "#555555")
            ax.text(0.20, y, f"{pid} ({btype_r})",
                    ha="left", va="top",
                    fontsize=9, color=name_color,
                    transform=ax.transAxes)

            # Hands survived
            ax.text(0.97, y, f"{hand_survived} hands",
                    ha="right", va="top",
                    fontsize=9, color="#888888",
                    transform=ax.transAxes)

    # ── Highlights feed ───────────────────────────────────────────────────────

    def _detect_highlights(self, hand_num: int):
        """Compare last two chip snapshots and append notable events."""
        if len(self.chip_history) < 2:
            return

        prev = self.chip_history[-2]
        curr = self.chip_history[-1]

        # Age existing highlights
        for h in self._highlights:
            h["age"] += 1

        deltas = {pid: curr.get(pid, 0) - prev.get(pid, 0)
                  for pid in self.player_ids}

        # Active players in each snapshot
        prev_active = [pid for pid in self.player_ids if prev.get(pid, 0) > 0]
        curr_active = [pid for pid in self.player_ids if curr.get(pid, 0) > 0]

        curr_leader = (max(curr_active, key=lambda p: curr.get(p, 0))
                       if curr_active else None)

        avg_stack = (sum(prev.get(p, 0) for p in prev_active) / len(prev_active)
                     if prev_active else self.starting_chips)

        new_entries     = []
        flagged_big_pot = set()
        flagged_double  = set()

        # 1. Eliminations
        for pid in self.player_ids:
            old_chips = prev.get(pid, 0)
            new_chips = curr.get(pid, 0)
            colour    = self.colours.get(pid, "white")
            if old_chips > 0 and new_chips <= 0:
                pos     = self._eliminations.get(pid, len(self.player_ids))
                ordinal = self._ordinal(pos)
                new_entries.append({
                    "hand": hand_num,
                    "text": f"#{hand_num} {pid} eliminated ({ordinal})",
                    "color": colour,
                    "age": 0,
                })
                flagged_big_pot.add(pid)
                flagged_double.add(pid)

        # 2. Chip leader change — compare against persistent tracker
        if (curr_leader is not None
                and curr_leader != self._last_chip_leader
                and self._last_chip_leader is not None):
            chips  = curr.get(curr_leader, 0)
            colour = self.colours.get(curr_leader, "white")
            new_entries.append({
                "hand": hand_num,
                "text": f"#{hand_num} {curr_leader} takes the lead! ({chips:,})",
                "color": colour,
                "age": 0,
            })
            flagged_big_pot.add(curr_leader)
            flagged_double.add(curr_leader)
        if curr_leader is not None:
            self._last_chip_leader = curr_leader

        # 3 & 4. Big pot / double-up (per player)
        threshold = self.starting_chips * 0.25
        for pid in self.player_ids:
            if pid in flagged_big_pot:
                continue
            delta     = deltas[pid]
            old_chips = prev.get(pid, 0)
            new_chips = curr.get(pid, 0)
            colour    = self.colours.get(pid, "white")

            # Big pot
            if delta > threshold:
                new_entries.append({
                    "hand": hand_num,
                    "text": f"#{hand_num} {pid} wins +{delta:,} chips!",
                    "color": colour,
                    "age": 0,
                })
                flagged_double.add(pid)
                continue

            # Double-up
            if pid in flagged_double:
                continue
            if (old_chips > 0
                    and old_chips < avg_stack * 0.35
                    and delta > 0
                    and new_chips >= old_chips * 1.8):
                new_entries.append({
                    "hand": hand_num,
                    "text": f"#{hand_num} {pid} doubles up! ({old_chips:,}→{new_chips:,})",
                    "color": colour,
                    "age": 0,
                })

        self._highlights.extend(new_entries)
        if len(self._highlights) > 30:
            self._highlights = self._highlights[-30:]

    def _draw_feed(self):
        """Redraw the highlights feed vertical sidebar."""
        ax = self.feed_ax
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Title
        ax.text(0.5, 0.97, "Highlights",
                ha="center", va="top",
                fontsize=11, fontweight="bold", color="white",
                transform=ax.transAxes)

        # Separator line under title
        ax.axhline(y=0.93, xmin=0.05, xmax=0.95,
                   color="#444", linewidth=0.8)

        if not self._highlights:
            ax.text(0.5, 0.50, "Waiting for\naction...",
                    ha="center", va="center",
                    fontsize=8, color="#555555", style="italic",
                    transform=ax.transAxes)
            return

        # Show the 8 most recent highlights, newest first (top to bottom)
        recent = list(reversed(self._highlights[-8:]))
        n      = len(recent)
        y_top  = 0.88
        y_bot  = 0.05
        step   = (y_top - y_bot) / max(n, 1)

        for i, entry in enumerate(recent):
            y     = y_top - i * step
            alpha = max(0.35, 1.0 - entry["age"] * 0.02)
            color = entry["color"]

            # Colored square dot
            ax.plot(0.07, y, "s",
                    color=color, markersize=5,
                    transform=ax.transAxes, clip_on=False,
                    alpha=alpha)

            # Text — left-aligned after dot
            ax.text(0.16, y, entry["text"],
                    ha="left", va="center",
                    fontsize=7.5, color=color,
                    alpha=alpha,
                    wrap=True,
                    transform=ax.transAxes)

    # ── Entry point ───────────────────────────────────────────────────────────

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Texas Hold'em tournament viewer")
    parser.add_argument("--players", type=str, default=DEFAULT_PLAYERS,
                        help=f"Comma-separated bot types (default: {DEFAULT_PLAYERS})")
    parser.add_argument("--chips", type=int, default=DEFAULT_CHIPS,
                        help=f"Starting chips (default: {DEFAULT_CHIPS})")
    parser.add_argument("--sb", type=int, default=DEFAULT_SMALLBLIND,
                        help=f"Starting small blind (default: {DEFAULT_SMALLBLIND})")
    parser.add_argument("--bb", type=int, default=DEFAULT_BIGBLIND,
                        help=f"Starting big blind (default: {DEFAULT_BIGBLIND})")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Initial delay between hands in seconds (default: {DEFAULT_DELAY})")
    parser.add_argument("--blind-increase-every", type=int,
                        default=DEFAULT_BLIND_INCREASE_EVERY,
                        help="Increase blinds 1.5x every N hands, 0 to disable (default: 50)")
    parser.add_argument("--rl_model", type=str, default=None,
                        help="Path to RL model weights (e.g. models/rl_model_run3.pt). "
                             "Rewrites any 'rl' entry in --players to use this model.")
    args = parser.parse_args()

    if args.rl_model:
        import re
        args.players = re.sub(r'(?<![:\w])rl(?![\w:])', f'rl:{args.rl_model}', args.players)

    players = parse_players(args.players)
    if len(players) < 2:
        print("Error: need at least 2 players. Check your --players spec.")
        return

    print(f"Players: {', '.join(f'{pid}={btype}' for pid, btype, _ in players)}")
    print(f"Chips: {args.chips}  |  Blinds: {args.sb}/{args.bb}  |  "
          f"Escalation every {args.blind_increase_every} hands")

    ui = TournamentUI(
        players=players,
        starting_chips=args.chips,
        base_sb=args.sb,
        base_bb=args.bb,
        hand_delay=args.delay,
        blind_increase_every=args.blind_increase_every,
    )
    ui.show()


if __name__ == "__main__":
    main()
