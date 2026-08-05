"""Latency figure for a battle, drawn on its own thread.

The gameplay loop is single threaded: it grabs a snapshot, blocks on the model,
then taps. Whatever that block costs is time the bot spends not looking at the
arena, and the placement it eventually makes answers a board that has already
moved on. None of that is visible in the log without reading every
`[Latency]` line by hand, so this draws it.

Rendering runs on a background thread for the same reason recording does: a
figure is worth nothing if it costs the loop a frame of reaction time. The
thread re-renders every few seconds rather than only at the end, so a run that
crashes or gets killed mid-match still leaves a usable picture behind.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import matplotlib

# Set before any pyplot-adjacent import: the interactive backends want the main
# thread, and this never draws to a screen.
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

# Validated for colorblind separation as an ordered set; keep the order.
COLOR_API = "#3f6fd8"
COLOR_CAPTURE = "#0f9488"
COLOR_GAP = "#d97706"
COLOR_PLACE = "#b4419e"

SURFACE = "#ffffff"
INK = "#000000"
INK_MUTED = "#3a3a3a"
GRID = "#d5d5d5"

# Journal-figure conventions: serif text, a closed axes box, inward ticks with
# minor divisions, and hairline rules. Colour still carries series identity, but
# hatching repeats it so the figure survives being printed in greyscale.
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": INK,
    "legend.fancybox": False,
    "hatch.linewidth": 0.6,
}


class FigureRecorder:
    """Collects per-decision timings and periodically redraws the figure."""

    def __init__(self, output_path: str, interval_sec: float = 10.0, title: str = ""):
        self.output_path = output_path
        self.interval_sec = interval_sec
        self.title = title

        self._lock = threading.Lock()
        self._samples: list[dict] = []
        self._events: list[tuple[float, str]] = []
        self._start = time.time()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dirty = False

    # -- collection ---------------------------------------------------------

    def record_decision(
        self,
        *,
        capture_latency_ms: float,
        api_latency_ms: float,
        action: str,
        card: Optional[str],
        executed: bool,
        elixir: Optional[float],
        usage: Optional[dict] = None,
    ) -> None:
        usage = usage or {}
        # The two providers name the cached-prefix counter differently; Claude
        # reports it at the top level, OpenAI nests it under input_tokens_details.
        cache_read = usage.get("cache_read_input_tokens")
        if cache_read is None:
            cache_read = (usage.get("input_tokens_details") or {}).get("cached_tokens")
        with self._lock:
            self._samples.append({
                "t": time.time() - self._start,
                "capture_s": (capture_latency_ms or 0.0) / 1000.0,
                "api_s": (api_latency_ms or 0.0) / 1000.0,
                "action": action,
                "card": card,
                "executed": executed,
                "elixir": elixir,
                "input_tokens": usage.get("input_tokens") or 0,
                "output_tokens": usage.get("output_tokens") or 0,
                "cache_read": cache_read or 0,
            })
            self._dirty = True

    def record_event(self, label: str) -> None:
        """Mark a moment worth a vertical line — match start, match end."""
        with self._lock:
            self._events.append((time.time() - self._start, label))
            self._dirty = True

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._start = time.time()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="figures")
        self._thread.start()

    def _loop(self) -> None:
        while self._running.is_set():
            # Poll in short slices so stop() does not wait out a whole interval.
            deadline = time.time() + self.interval_sec
            while self._running.is_set() and time.time() < deadline:
                time.sleep(0.25)
            if not self._running.is_set():
                break
            with self._lock:
                dirty = self._dirty
            if dirty:
                self._render()

    def stop(self, timeout: float = 5.0) -> Optional[str]:
        """Draw the final figure and return its path, or None if nothing to draw."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        with self._lock:
            empty = not self._samples
        if empty:
            return None
        return self._render()

    # -- drawing ------------------------------------------------------------

    def _render(self) -> Optional[str]:
        with self._lock:
            samples = list(self._samples)
            events = list(self._events)
            elapsed = time.time() - self._start
            self._dirty = False
        if not samples:
            return None

        with matplotlib.rc_context(PAPER_RC):
            fig = Figure(figsize=(7.2, 8.6), dpi=220, facecolor=SURFACE)
            FigureCanvasAgg(fig)
            grid = fig.add_gridspec(
                3, 1, height_ratios=[1.15, 1.0, 0.8],
                left=0.115, right=0.965, top=0.955, bottom=0.115, hspace=0.34,
            )
            ax_latency = fig.add_subplot(grid[0])
            ax_elixir = fig.add_subplot(grid[1])
            ax_tokens = fig.add_subplot(grid[2])
            for ax in (ax_latency, ax_elixir, ax_tokens):
                self._style_axes(ax)

            span = max(s["t"] for s in samples)
            self._draw_latency(ax_latency, samples, events, span)
            self._draw_elixir(ax_elixir, samples, events, span)
            self._draw_tokens(ax_tokens, samples)
            # Panel labels sit outside the axes box, so a legend can own the
            # strip directly above the plot without colliding with them.
            for ax, label in ((ax_latency, "(a)"), (ax_elixir, "(b)"), (ax_tokens, "(c)")):
                ax.text(-0.105, 1.10, label, transform=ax.transAxes,
                        fontsize=10, fontweight="bold", va="top", ha="left")

            self._draw_caption(fig, samples, elapsed)

            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            fig.savefig(self.output_path, facecolor=SURFACE)
        return self.output_path

    def _style_axes(self, ax) -> None:
        ax.set_facecolor(SURFACE)
        # Horizontal rules only, hairline and dotted: enough to read a value off
        # the axis without competing with the marks.
        ax.grid(True, axis="y", color=GRID, linewidth=0.5, linestyle=":", zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="both", labelsize=8)

    def _legend(self, ax, ncols=3):
        """One legend strip above each panel, clear of both marks and panel label."""
        return ax.legend(
            fontsize=7.5, borderpad=0.4, handlelength=1.6, labelspacing=0.3,
            columnspacing=1.4, frameon=False, ncols=ncols,
            loc="lower left", bbox_to_anchor=(0.0, 1.005),
        )

    def _draw_caption(self, fig, samples, elapsed) -> None:
        """Two lines: what each panel plots, then the run's numbers.

        The axes are already labelled, so the caption names each panel rather
        than describing it again.
        """
        api = sorted(s["api_s"] for s in samples)
        # Model call only. Capture runs in its own subprocess alongside the
        # loop, so counting it here would charge the loop for time it did not
        # actually spend waiting — enough to push the figure over 100%.
        blocked = sum(s["api_s"] for s in samples)
        placements = sum(1 for s in samples if s["executed"])
        cached = sum(1 for s in samples if s["cache_read"] > 0)

        def pct(values, q):
            if not values:
                return 0.0
            return values[min(len(values) - 1, int(q * len(values)))]

        caption = (
            "Figure 1. Per-decision timing, one battle. "
            "(a) decision time budget and gap since the previous decision; "
            "(b) elixir at decision time; (c) prompt tokens.\n"
            f"$n$ = {len(samples)} decisions, {placements} placements, {elapsed:.0f} s. "
            f"Model call median {pct(api, 0.5):.2f} s, p95 {pct(api, 0.95):.2f} s. "
            f"Loop blocked {blocked / elapsed * 100 if elapsed else 0:.0f}%. "
            f"Cache hits {cached}/{len(samples)}."
        )
        if self.title:
            caption += f" {self.title}."
        fig.text(0.115, 0.078, caption, fontsize=7.6, color=INK, va="top", ha="left",
                 wrap=True, linespacing=1.5)

    def _mark_events(self, ax, events, span) -> None:
        for t, label in events:
            ax.axvline(t, color=INK_MUTED, linewidth=0.7, linestyle=(0, (5, 3)), zorder=1)
            # A marker near the right edge has to label itself leftwards, or the
            # text runs off the canvas.
            late = span > 0 and t > span * 0.75
            ax.annotate(
                label, xy=(t, 0.02), xycoords=("data", "axes fraction"),
                xytext=(-4 if late else 4, 4), textcoords="offset points",
                ha="right" if late else "left", va="bottom", rotation=90,
                color=INK_MUTED, fontsize=6.5, style="italic",
            )

    @staticmethod
    def _bar_width(times) -> float:
        if len(times) < 2:
            return 1.0
        return max(0.6, (times[-1] - times[0]) / len(times) * 0.5)

    def _draw_latency(self, ax, samples, events, span) -> None:
        """Where each decision's seconds went, and how long the arena went unanswered."""
        times = [s["t"] for s in samples]
        capture = [s["capture_s"] for s in samples]
        api = [s["api_s"] for s in samples]
        # Gap from one decision to the next: the interval no snapshot is being
        # evaluated, which is the number that actually loses towers.
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        width = self._bar_width(times)

        ax.bar(times, capture, width=width, facecolor=COLOR_CAPTURE, edgecolor=INK,
               linewidth=0.5, label="capture", zorder=3)
        ax.bar(times, api, width=width, bottom=capture, facecolor=COLOR_API,
               edgecolor=INK, linewidth=0.5, hatch="///", label="model call", zorder=3)
        if gaps:
            ax.plot(
                times[1:], gaps, color=COLOR_GAP, linewidth=1.1, marker="s", markersize=3.4,
                markerfacecolor=SURFACE, markeredgewidth=0.9,
                label="interval since previous decision", zorder=4,
            )
            worst = max(range(len(gaps)), key=lambda i: gaps[i])
            ax.annotate(
                f"max {gaps[worst]:.1f} s", xy=(times[worst + 1], gaps[worst]),
                xytext=(0, 6), textcoords="offset points", ha="center",
                fontsize=7, color=INK,
            )

        self._mark_events(ax, events, span)
        ax.set_ylabel("time (s)", fontsize=9)
        ax.set_xlabel("match time (s)", fontsize=9)
        # Headroom so the max-gap annotation does not ride the top spine.
        peak = max(gaps + [c + a for c, a in zip(capture, api)] or [1.0])
        ax.set_ylim(0, peak * 1.15)
        self._legend(ax, ncols=3)

    def _draw_elixir(self, ax, samples, events, span) -> None:
        """Elixir at decision time, with what was actually played on top of it."""
        times = [s["t"] for s in samples]
        elixir = [s["elixir"] if s["elixir"] is not None else float("nan") for s in samples]
        ax.plot(times, elixir, color=INK_MUTED, linewidth=0.9, zorder=2)

        placed = [s for s in samples if s["executed"]]
        idles = [s for s in samples if not s["executed"]]
        if placed:
            ax.scatter(
                [s["t"] for s in placed],
                [s["elixir"] if s["elixir"] is not None else 0 for s in placed],
                s=26, marker="o", facecolor=COLOR_PLACE, edgecolor=INK, linewidth=0.5,
                zorder=5, label="card placed",
            )
            for s in placed:
                ax.annotate(
                    s["card"] or "?",
                    xy=(s["t"], s["elixir"] if s["elixir"] is not None else 0),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK, style="italic",
                )
        if idles:
            ax.scatter(
                [s["t"] for s in idles],
                [s["elixir"] if s["elixir"] is not None else 0 for s in idles],
                s=24, marker="^", facecolor=SURFACE, edgecolor=INK, linewidth=0.7,
                zorder=4, label="idle or not executed",
            )

        self._mark_events(ax, events, span)
        ax.set_ylim(0, 11)
        ax.set_ylabel("elixir", fontsize=9)
        ax.set_xlabel("match time (s)", fontsize=9)
        self._legend(ax, ncols=2)

    def _draw_tokens(self, ax, samples) -> None:
        """Prompt size per call, and how much of it the provider served from cache."""
        times = [s["t"] for s in samples]
        fresh = [max(0, s["input_tokens"] - s["cache_read"]) for s in samples]
        cached = [s["cache_read"] for s in samples]
        width = self._bar_width(times)

        ax.bar(times, fresh, width=width, facecolor=COLOR_API, edgecolor=INK,
               linewidth=0.5, hatch="///", label="input, billed fresh", zorder=3)
        if any(cached):
            ax.bar(times, cached, width=width, bottom=fresh, facecolor=COLOR_CAPTURE,
                   edgecolor=INK, linewidth=0.5, hatch="...", label="input, cache read",
                   zorder=3)
        ax.set_ylabel("tokens", fontsize=9)
        ax.set_xlabel("match time (s)", fontsize=9)
        ax.set_ylim(bottom=0)
        self._legend(ax, ncols=2)


def figure_path_for(record_path: Optional[str], figures_dir: str = "figures") -> str:
    """Name the figure after the recording, so a run's video and chart pair up."""
    if record_path:
        stem = os.path.splitext(os.path.basename(record_path))[0]
    else:
        stem = time.strftime("battle_%Y%m%d_%H%M%S")
    return os.path.join(figures_dir, f"{stem}.png")
