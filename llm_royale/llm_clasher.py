#!/usr/bin/env python3
"""
Run Clash capture, call an LLM for the next move, and execute the action on the iPhone.
"""

import argparse
import json
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from typing import Optional

import requests

from .mac_action import MirrorActionExecutor
from .match_navigator import BATTLE_BUTTON, BATTLE_TAB, OK_BUTTON
from .overlay import draw_overlay
from .recorder import VideoRecorder
from .stopper import DEFAULT_STOP_FILE, Stopper
from .capture_config import PRINCESS_TOWER_FULL_HP, KING_TOWER_FULL_HP
from .cycle_tracker import CycleTracker


CARD_COSTS = {
    "hog-rider": 4,
    "musketeer": 4,
    "ice-spirit": 1,
    "ice-golem": 2,
    "fireball": 4,
    "the-log": 2,
    "cannon": 3,
    "skeleton": 1,
}

# Clash Royale match timing (standard 1v1 ladder).
# Total match length: 5 minutes base.  Double elixir starts at 2:00 remaining
# (i.e. 3 min into the match), triple elixir at 1:00 remaining (4 min in).
# Overtime adds 3 more minutes, all at triple.
SINGLE_ELIXIR_RATE = 2.8   # seconds per 1 elixir
DOUBLE_ELIXIR_RATE = 1.4
TRIPLE_ELIXIR_RATE = 0.93
DOUBLE_ELIXIR_AT_SEC = 180.0   # 3 minutes into the match
TRIPLE_ELIXIR_AT_SEC = 240.0   # 4 minutes into the match

# How often to report that the loop is still waiting for a battle.
WAITING_REPORT_SEC = 5.0

# Consecutive snapshots without a battle before the match counts as over.
MATCH_END_CONFIRM_SNAPSHOTS = 6

# Elixir readings around a placement, used to tell whether the game actually
# accepted the card. Verification by detection cannot answer that on its own:
# spells leave no unit behind, and small troops are easy for the detector to
# miss entirely.
#
# Comparing consecutive snapshots is too coarse to answer it either — elixir
# regenerates between them, the reading the planner acted on is several hundred
# ms stale, and other placements land in between; that produced impossible
# figures like a 2-cost card apparently costing 5. The --verify-elixir mode
# instead reads the bar directly either side of the swipe, which is exact but
# adds this settle time to every placement, so it is a diagnostic and not the
# default.
ELIXIR_DEBIT_TOLERANCE = 0.6
ELIXIR_SETTLE_AFTER_PLAY_SEC = 0.55


class TowerHealthTracker:
    """Smooths per-tower HP readings across snapshots.

    A troop standing in front of a tower hides part of its HP number, and the
    OCR then reports something like 94 for a tower actually sitting at 2254.
    Tower HP only ever falls during a match, and never by thousands between two
    snapshots, so a reading that violates both has to be confirmed by a second
    snapshot before it is believed. That costs one snapshot of latency on a real
    burst of damage and rejects the misreads outright.

    An increase is treated the same way, which is what lets a new match through:
    the towers come back at full HP and stay there, so the second snapshot
    confirms it.
    """

    MAX_BELIEVABLE_DROP = 1200
    CONFIRM_TOLERANCE = 0.10

    def __init__(self):
        self.accepted = {}
        self.pending = {}

    def _confirms(self, name: str, value: int) -> bool:
        previous = self.pending.get(name)
        self.pending[name] = value
        if previous is None:
            return False
        tolerance = max(50.0, abs(previous) * self.CONFIRM_TOLERANCE)
        return abs(previous - value) <= tolerance

    def update(self, towers: dict) -> dict:
        for name, tower in (towers or {}).items():
            value = tower.get("value")
            if value is None:
                continue
            last = self.accepted.get(name)

            plausible = last is None or (value <= last and last - value <= self.MAX_BELIEVABLE_DROP)
            if plausible:
                self.accepted[name] = value
                self.pending.pop(name, None)
            elif self._confirms(name, value):
                self.accepted[name] = value
                self.pending.pop(name, None)
            else:
                # Keep the last trusted number and mark it, so the planner (and
                # anyone reading the log) knows this one is held over.
                tower["value"] = last
                tower["status"] = "held_pending_confirmation"
                tower["rejected_reading"] = value
        return towers


class ElixirClock:
    """
    Soft elixir tracker that auto-switches regen rate based on game phase.

    Clash Royale regen rates:
        1x elixir: 1 elixir per 2.8s  (0:00 – 3:00)
        2x elixir: 1 elixir per 1.4s  (3:00 – 4:00)
        3x elixir: 1 elixir per 0.93s (4:00+, including overtime)
    """

    def __init__(self, start: float = 5.0, max_elixir: float = 10.0):
        self.value = float(start)
        self.max_elixir = float(max_elixir)
        self.last_update = time.time()
        self.game_start: Optional[float] = None
        self.started = False

    def start(self) -> None:
        self.value = 5.0
        self.last_update = time.time()
        self.game_start = time.time()
        self.started = True

    @property
    def elapsed_sec(self) -> float:
        if self.game_start is None:
            return 0.0
        return time.time() - self.game_start

    @property
    def game_phase(self) -> str:
        elapsed = self.elapsed_sec
        if elapsed >= TRIPLE_ELIXIR_AT_SEC:
            return "3x"
        if elapsed >= DOUBLE_ELIXIR_AT_SEC:
            return "2x"
        return "1x"

    @property
    def rate_sec_per_elixir(self) -> float:
        phase = self.game_phase
        if phase == "3x":
            return TRIPLE_ELIXIR_RATE
        if phase == "2x":
            return DOUBLE_ELIXIR_RATE
        return SINGLE_ELIXIR_RATE

    def advance(self) -> float:
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        if self.started:
            self.value = min(self.max_elixir, self.value + dt / self.rate_sec_per_elixir)
        return self.value

    def spend(self, card: str) -> None:
        cost = CARD_COSTS.get(card)
        if cost is None:
            return
        self.advance()
        self.value = max(0.0, self.value - cost)

    def estimate(self, observed: Optional[int]) -> float:
        """
        Blend our internal clock with any vision-based reading. If the vision
        reading is present and plausible, prefer it (snap to it) and resync
        the clock; otherwise return the clock.
        """
        current = self.advance()
        if observed is not None and 0 <= observed <= 10:
            if abs(observed - current) >= 2.0:
                # Vision claims something very different — trust it and resync.
                self.value = float(observed)
            return float(self.value)
        return round(current, 1)


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def wait_for_space() -> None:
    if not sys.stdin.isatty():
        # Unattended runs (background shells, CI, cron) have no terminal to read
        # a keypress from; starting straight away is the only sane behaviour.
        print("[INFO] stdin is not a terminal, starting immediately.")
        return
    print("[INFO] Press SPACE to start the capture/LLM loop. Press Ctrl-C to quit.")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable:
                ch = sys.stdin.read(1)
                if ch == " ":
                    print("\n[INFO] Starting...")
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def format_ai_decision(decision: dict) -> str:
    action = str(decision.get("action", "idle")).lower()

    if action == "place_card":
        parts = [f"play {decision.get('card', '?')}"]
        x_norm = decision.get("x_norm")
        y_norm = decision.get("y_norm")
        if x_norm is not None and y_norm is not None:
            parts.append(f"at ({float(x_norm):.2f}, {float(y_norm):.2f})")
        elif decision.get("x") is not None and decision.get("y") is not None:
            parts.append(f"at ({float(decision['x']):.0f}, {float(decision['y']):.0f})")
        return " ".join(parts)

    return action


def format_ai_result(result: dict) -> str:
    status = str(result.get("status", "unknown"))
    reason = str(result.get("reason", "")).strip()

    if status == "executed":
        card = result.get("card", "?")
        slot = result.get("slot", "?")
        target = result.get("capture_target", {}) or {}
        x_norm = target.get("x_norm")
        y_norm = target.get("y_norm")
        summary = f"executed {card} from slot {slot}"
        if x_norm is not None and y_norm is not None:
            summary += f" to ({float(x_norm):.2f}, {float(y_norm):.2f})"
        if reason:
            summary += f" | {reason}"
        return summary

    if reason:
        return f"{status}: {reason}"
    return status


class CaptureWorker:
    def __init__(self, python_bin: str, output_json: str):
        self.python_bin = python_bin
        self.output_json = output_json
        self.thread = None
        self.proc = None
        self.stop_event = threading.Event()
        self.exit_code = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        cmd = [
            self.python_bin,
            # Same reason as run.sh: its stdout is a pipe, so without this its
            # per-snapshot lines sit in a buffer and never reach the log.
            "-u",
            "-m",
            "llm_royale.clash_capture",
            "--preset",
            "fast",
            "--output-json",
            self.output_json,
            # 1.0s → 0.4s. The previous cadence meant the LLM saw the board
            # up to a second stale, and a Hog Rider crosses the bridge in ~1.5s.
            "--interval-sec",
            "0.4",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            print(f"[capture] {line.rstrip()}")
            if self.stop_event.is_set():
                break
        self.exit_code = self.proc.wait()

    def stop(self) -> None:
        self.stop_event.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def failed(self) -> bool:
        return self.exit_code not in (None, 0)


class Planner:
    """Everything about deciding a move that is not the API call itself.

    Prompt construction, the board summary, the decision schema, and the
    validation of what comes back are identical whichever model is asked —
    only `request_decision_json` differs per provider. Keeping the split here
    means a change to how the board is described cannot drift between them.
    """

    # Provider SDK exceptions the main loop should ride out rather than die on.
    # Empty here because the OpenAI planner raises through `requests`, which the
    # loop already handles by name.
    transient_api_errors: tuple = ()

    def __init__(self, model: str):
        self.model = model
        self.last_api_debug = {}
        self.recent_api_latencies_ms = deque(maxlen=8)
        self.recent_actions: deque = deque(maxlen=4)

    def request_decision_json(self, *, instructions: str, input_text: str) -> str:
        """Ask the model for one decision. Returns the raw JSON text."""
        raise NotImplementedError

    def estimate_decision_latency_sec(self, snapshot: dict) -> float:
        capture_latency_ms = float(snapshot.get("capture", {}).get("total_latency_ms", 0.0) or 0.0)
        if self.recent_api_latencies_ms:
            api_latency_ms = sum(self.recent_api_latencies_ms) / len(self.recent_api_latencies_ms)
        else:
            api_latency_ms = 1200.0
        action_overhead_ms = 250.0
        total_ms = capture_latency_ms + api_latency_ms + action_overhead_ms
        return round(total_ms / 1000.0, 2)

    def build_compact_snapshot(self, snapshot: dict) -> dict:
        hand_slots = snapshot.get("hand_cards_inferred", {}).get("slots", [])
        compact_hand = [
            {"slot": slot.get("slot"), "label": slot.get("label")}
            for slot in hand_slots
        ]
        board_reference = snapshot.get("llm_summary", {}).get("board_reference", {})
        decision_latency_sec = self.estimate_decision_latency_sec(snapshot)

        def pick(items, limit=5):
            out = []
            for item in items[:limit]:
                out.append({
                    "label": item.get("label"),
                    "owner": item.get("owner"),
                    "lane": item.get("lane"),
                    "side": item.get("side"),
                    "x": item.get("center", {}).get("x"),
                    "y": item.get("center", {}).get("y"),
                    "x_norm": item.get("center_normalized", {}).get("x"),
                    "y_norm": item.get("center_normalized", {}).get("y"),
                })
            return out

        def pick_threat(items, limit=4):
            return [
                {
                    "label": u.get("label"),
                    "lane": u.get("lane"),
                    "pressure": u.get("pressure_score"),
                    "x_norm": u.get("center_normalized", {}).get("x"),
                    "y_norm": u.get("center_normalized", {}).get("y"),
                }
                for u in items[:limit]
            ]

        llm_summary = snapshot.get("llm_summary", {})
        # ONLY the currently-visible hand is playable. The cycle tracker's
        # smoothed view caused the LLM to ask for cards that weren't in the
        # hand, which with the old fallback executor caused the wrong card to
        # be played (e.g. asking for `the-log` and placing an ice-golem).
        playable_cards = sorted({s["label"] for s in compact_hand if s.get("label")})
        playable_with_cost = [
            {"card": c, "cost": CARD_COSTS.get(c)} for c in playable_cards
        ]

        elixir_inferred = snapshot.get("elixir_inferred", {}) or {}
        elixir_value = elixir_inferred.get("blended_estimate")
        if elixir_value is None:
            elixir_value = elixir_inferred.get("count_estimate")

        # --- Tower health as percentage (0-100) WITH positions ---
        tower_health_raw = llm_summary.get("tower_health", [])
        towers = {}
        for t in tower_health_raw:
            owner = t.get("owner")  # "friendly" or "enemy"
            label = t.get("label")  # "queen-tower" or "king-tower"
            lane = t.get("lane")
            hp = t.get("hp_estimate")
            ratio = t.get("health_ratio_estimate")

            if label == "king-tower":
                key = f"{owner}_king"
                max_hp = KING_TOWER_FULL_HP
            elif lane in ("left", "right"):
                key = f"{owner}_{lane}"
                max_hp = PRINCESS_TOWER_FULL_HP
            else:
                continue

            # Prefer the OCR hp_estimate; fall back to ratio from bar fill.
            if hp is not None and max_hp > 0:
                pct = round(hp / max_hp * 100)
            elif ratio is not None:
                pct = round(ratio * 100)
            else:
                pct = None

            # Keep the first (highest-confidence) entry per key.
            if key not in towers and pct is not None:
                towers[key] = {"hp": pct}

        # Attach tower positions from grouped detections (these have center_normalized).
        tower_detections = snapshot.get("grouped", {}).get("towers", [])
        for td in tower_detections:
            td_owner = td.get("owner")
            td_label = td.get("label")
            td_lane = td.get("lane")
            cn = td.get("center_normalized", {})
            if td_label == "king-tower":
                td_key = f"{td_owner}_king"
            elif td_lane in ("left", "right"):
                td_key = f"{td_owner}_{td_lane}"
            else:
                continue
            if td_key in towers:
                towers[td_key]["x"] = cn.get("x")
                towers[td_key]["y"] = cn.get("y")
            elif cn.get("x") is not None:
                # Tower detected but no HP — still include position.
                towers[td_key] = {"hp": None, "x": cn.get("x"), "y": cn.get("y")}

        # --- Cycle state (what the LLM can see coming next) ---
        cycle_state = snapshot.get("cycle_state", {})
        cycle_info = None
        if cycle_state:
            played = list(cycle_state.get("played_history", []))
            cycle_info = {
                "last_4_played": played[-4:] if played else [],
                "next_cycle_candidates": cycle_state.get("next_cycle_candidates", []),
            }

        # --- Game phase ---
        game_phase = snapshot.get("_game_phase", "1x")

        # --- Recent action history (what we already did) ---
        recent = list(self.recent_actions)

        return {
            "seq": snapshot.get("sequence"),
            "w": snapshot.get("capture", {}).get("width"),
            "h": snapshot.get("capture", {}).get("height"),
            "decision_latency_sec": decision_latency_sec,
            "game_phase": game_phase,
            "board": {
                "river_y_norm": board_reference.get("river_y_norm"),
                "lane_left_norm": board_reference.get("lane_left_boundary_norm"),
                "lane_right_norm": board_reference.get("lane_right_boundary_norm"),
                "place_bbox_norm": board_reference.get("troop_placement_bbox_norm"),
            },
            "elixir": elixir_value,
            "towers": towers,
            "hand": compact_hand,
            "playable_cards": playable_cards,
            "playable_with_cost": playable_with_cost,
            "cycle": cycle_info,
            "recent_actions": recent,
            "enemy_on_our_side": pick_threat(llm_summary.get("enemy_units_on_friendly_side", [])),
            "friendly_on_enemy_side": pick_threat(llm_summary.get("friendly_units_on_enemy_side", [])),
            "lane_balance": llm_summary.get("lane_balance", {}),
            "top_threats": pick_threat(llm_summary.get("top_threats", [])),
            "friendly_troops": pick(snapshot.get("grouped", {}).get("troops", [])),
            "friendly_buildings": pick(snapshot.get("grouped", {}).get("buildings", [])),
        }

    def build_instructions(self) -> str:
        """The static half of every request.

        Deliberately free of strategy: the model is told the deck, the frame of
        reference, what the fields mean, and what it is not allowed to do, then
        left to work out the play itself. Two reasons. The old prompt's tactics
        were a fixed script that could not answer a board it had not anticipated,
        and every line of it was resent on every call — at roughly one decision
        every four seconds, prompt length is latency.

        Nothing here varies between calls, which is the point: an unchanging
        system prompt is a cacheable prefix. Everything live, including the
        control lag, travels in the user message.
        """
        return (
            "You play Clash Royale. Your deck: hog-rider(4), musketeer(4), "
            "ice-spirit(1), ice-golem(2), fireball(4), the-log(2), cannon(3), skeleton(1). "
            "Costs in parentheses are elixir.\n"
            "\n"
            "COORDINATES\n"
            "Normalized 0-1 over the arena. y=0 is the enemy king end, y increases toward you. "
            "`board.river_y_norm` is the river; y above it is enemy territory, y below it is yours. "
            "Placeable region is `board.place_bbox_norm`. Left lane x≈0.22, center x≈0.50, right x≈0.78.\n"
            "\n"
            "INPUT FIELDS\n"
            "- `elixir`: your current elixir, 0-10.\n"
            "- `game_phase`: 1x, 2x or 3x elixir generation rate.\n"
            "- `hand`: your four slots; a null label is a card the vision could not read, not an empty slot.\n"
            "- `playable_cards` / `playable_with_cost`: the cards you may legally play right now.\n"
            "- `towers`: each tower's `hp` (0-100%) and detected `x`,`y`.\n"
            "- `enemy_on_our_side`, `friendly_on_enemy_side`, `friendly_troops`, `friendly_buildings`, "
            "`top_threats`: detected units with `x_norm`,`y_norm`.\n"
            "- `lane_balance`: unit counts and pressure per lane.\n"
            "- `cycle`: cards recently played and what is coming back.\n"
            "- `recent_actions`: your own last few plays.\n"
            "- `decision_latency_sec`: see STALE STATE.\n"
            "\n"
            "STALE STATE\n"
            "The snapshot was captured `decision_latency_sec` seconds ago and your placement lands "
            "that much later still. Units have moved since. Lead moving targets, and treat a threat's "
            "listed position as where it was, not where it is.\n"
            "\n"
            "YOUR TASK\n"
            "Choose the action that maximizes your probability of winning from this state. "
            "Decide the strategy yourself — no playstyle is prescribed. Idle is a real move; "
            "take it when no placement beats holding elixir.\n"
            "\n"
            "CONSTRAINTS\n"
            "- Play only a card listed in `playable_cards`. If it is empty, idle.\n"
            "- Never play a card costing more than `elixir`.\n"
            "- Do not replay a card your `recent_actions` show you just played, unless the board "
            "makes repeating it clearly correct.\n"
            "- x_norm and y_norm must fall inside `board.place_bbox_norm`.\n"
            "- Spells must target a position where a unit or tower actually appears in the snapshot.\n"
            "- For idle, set card, x_norm and y_norm to null.\n"
            "\n"
            "OUTPUT\n"
            "Strict JSON only, no prose:\n"
            '{"action":"idle","card":null,"x_norm":null,"y_norm":null}\n'
            '{"action":"place_card","card":"<name>","x_norm":<0..1>,"y_norm":<0..1>}'
        )

    def build_input(self, snapshot: dict) -> str:
        compact = self.build_compact_snapshot(snapshot)
        return json.dumps(compact, separators=(",", ":"))

    def decision_output_schema(self) -> dict:
        cards = sorted(CARD_COSTS)
        return {
            "type": "json_schema",
            "name": "clash_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {"type": "string", "enum": ["idle", "place_card"]},
                    "card": {"type": ["string", "null"], "enum": cards + [None]},
                    "x_norm": {"type": ["number", "null"]},
                    "y_norm": {"type": ["number", "null"]},
                },
                "required": ["action", "card", "x_norm", "y_norm"],
            },
        }

    def _extract_output_text(self, data: dict) -> str:
        if data.get("status") == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason", "unknown")
            raise ValueError(f"Model response incomplete: {reason}")

        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "refusal":
                    raise ValueError(f"Model refusal: {content.get('refusal', 'unspecified refusal')}")

        text = data.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        chunks = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") != "output_text":
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        joined = "".join(chunks).strip()
        if not joined:
            raise ValueError("Empty model response")
        return joined

    def _decision_from_text(self, text: str) -> dict:
        decision = json.loads(text)
        action = decision.get("action")
        if action == "idle":
            decision["card"] = None
            decision["x_norm"] = None
            decision["y_norm"] = None
        elif action == "place_card":
            if decision.get("card") is None or decision.get("x_norm") is None or decision.get("y_norm") is None:
                raise ValueError("place_card decision missing card or coordinates")
        else:
            raise ValueError(f"Unsupported action in model response: {action}")
        return decision

    def _responses_create(self, *, instructions: str, input_text: str) -> dict:
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "store": False,
            "text": {
                "format": self.decision_output_schema(),
            },
            "temperature": 0,
            # Structured output lets us tighten the cap; shorter responses
            # reduce latency in the live control loop.
            "max_output_tokens": 120,
        }
        if self.effort is not None:
            payload["reasoning"] = {"effort": self.effort}
        start = time.time()
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        elapsed_ms = round((time.time() - start) * 1000.0, 2)
        self.last_api_debug = {
            "model": self.model,
            "provider": "openai",
            "effort": self.effort,
            "http_status": response.status_code,
            "latency_ms": elapsed_ms,
            "input_bytes": len(input_text.encode("utf-8")),
            "instructions_chars": len(instructions),
        }
        if not response.ok:
            body = response.text[:1000]
            raise requests.HTTPError(
                f"{response.status_code} error from Responses API: {body}",
                response=response,
            )
        data = response.json()
        self.last_api_debug["response_status"] = data.get("status")
        if data.get("incomplete_details"):
            self.last_api_debug["incomplete_details"] = data["incomplete_details"]
        usage = data.get("usage", {})
        if usage:
            self.last_api_debug["usage"] = usage
        self.recent_api_latencies_ms.append(elapsed_ms)
        return data

    def plan(self, snapshot: dict) -> dict:
        text = self.request_decision_json(
            instructions=self.build_instructions(),
            input_text=self.build_input(snapshot),
        )
        return self._decision_from_text(text)


class OpenAIPlanner(Planner):
    """Planner backed by the OpenAI Responses API."""

    def __init__(self, model: str, effort: Optional[str] = None):
        super().__init__(model)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai provider")
        # Left off the request entirely unless asked for. Not every model takes
        # a reasoning effort, and sending one by default would change how the
        # existing setup behaves for the sake of a knob nobody set.
        # Left off the request entirely unless asked for. Not every model takes
        # a reasoning effort, and sending one by default would change how the
        # existing setup behaves for the sake of a knob nobody set.
        self.effort = effort

    def request_decision_json(self, *, instructions: str, input_text: str) -> str:
        return self._extract_output_text(
            self._responses_create(instructions=instructions, input_text=input_text)
        )


class AnthropicPlanner(Planner):
    """Planner backed by the Claude Messages API.

    Two settings here are about the control loop, not about capability. Thinking
    is disabled and effort is low because this runs live against a match: a
    placement is worth nothing if it arrives after the push it was answering
    has already crossed the bridge. Structured outputs then guarantee the reply
    parses as the decision schema, which also removes the one real hazard of
    running Claude with thinking off — reasoning leaking into the text as
    unparseable prose.
    """

    # Enough for the decision object; the schema keeps responses short.
    MAX_TOKENS = 256

    # Room for the model to think when effort demands it; the schema still keeps
    # the visible answer short.
    MAX_TOKENS_THINKING = 4096

    def __init__(self, model: str, effort: str = "low"):
        super().__init__(model)
        if effort not in EFFORT_LEVELS:
            raise RuntimeError(
                f"unknown effort {effort!r}; choose one of {', '.join(EFFORT_LEVELS)}"
            )
        self.effort = effort
        self.thinking = effort in EFFORT_REQUIRING_THINKING
        # Older models reject `effort` outright rather than ignoring it, which
        # kills the run on the first snapshot. Drop the parameter for those and
        # say so, instead of asking the caller to remember which models take it.
        self.supports_effort = self.model not in MODELS_WITHOUT_EFFORT
        if not self.supports_effort:
            print(f"[INFO] {self.model} does not accept an effort level; sending the request without one")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for the anthropic provider")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "The anthropic package is missing. Run ./install.sh, or "
                "pip install anthropic."
            ) from e
        self._client = anthropic.Anthropic()
        # Handed to the main loop so one bad call idles a tick instead of
        # ending the match: the SDK raises its own exception types, which
        # nothing in the loop's except clauses would otherwise match.
        self.transient_api_errors = (anthropic.APIStatusError, anthropic.APIConnectionError)

    def decision_schema(self) -> dict:
        """The decision shape, in the schema dialect structured outputs accepts.

        Same fields as the OpenAI schema, but nullable values are spelled as an
        anyOf rather than a type union — the union form is not part of the
        supported subset.
        """
        cards = sorted(CARD_COSTS)
        nullable_number = {"anyOf": [{"type": "number"}, {"type": "null"}]}
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["idle", "place_card"]},
                "card": {"anyOf": [{"type": "string", "enum": cards}, {"type": "null"}]},
                "x_norm": nullable_number,
                "y_norm": nullable_number,
            },
            "required": ["action", "card", "x_norm", "y_norm"],
        }

    def request_decision_json(self, *, instructions: str, input_text: str) -> str:
        output_config = {"format": {"type": "json_schema", "schema": self.decision_schema()}}
        if self.supports_effort:
            output_config["effort"] = self.effort
        started = time.time()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.MAX_TOKENS_THINKING if self.thinking else self.MAX_TOKENS,
            system=instructions,
            messages=[{"role": "user", "content": input_text}],
            thinking={"type": "adaptive"} if self.thinking else {"type": "disabled"},
            output_config=output_config,
        )
        elapsed_ms = round((time.time() - started) * 1000.0, 2)

        usage = response.usage
        self.last_api_debug = {
            "model": self.model,
            "provider": "anthropic",
            "effort": self.effort if self.supports_effort else None,
            "thinking": self.thinking,
            "latency_ms": elapsed_ms,
            "input_bytes": len(input_text.encode("utf-8")),
            "instructions_chars": len(instructions),
            "response_status": response.stop_reason,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            },
        }
        self.recent_api_latencies_ms.append(elapsed_ms)

        # Check why generation stopped before reading content: a refusal comes
        # back as a normal 200 with no usable text, and max_tokens leaves the
        # JSON truncated — both would otherwise surface as a parse error.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise ValueError(f"Claude refused the request ({getattr(details, 'category', 'unspecified')})")
        if response.stop_reason == "max_tokens":
            raise ValueError("Claude response hit max_tokens before completing the decision")

        for block in response.content:
            if block.type == "text" and block.text.strip():
                return block.text.strip()
        raise ValueError("Claude response contained no text block")


PLANNERS = {
    "openai": OpenAIPlanner,
    "anthropic": AnthropicPlanner,
}

# Per-provider defaults, used when --model is not given.
DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-opus-5",
}

# Effort levels both providers accept, cheapest and fastest first. OpenAI also
# takes "none" and "minimal"; those are left out because Claude rejects them and
# one shared vocabulary is worth more here than the extra two rungs.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Claude runs at this when no effort is given: the live loop wants speed, and
# anything above `high` forces thinking on.
DEFAULT_ANTHROPIC_EFFORT = "low"

# Claude models that reject `effort` with a 400 rather than ignoring it. The
# parameter arrived with Opus 4.5, so anything older than that predates it.
MODELS_WITHOUT_EFFORT = {
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-0",
    "claude-opus-4-1",
    "claude-opus-4-0",
}

# Thinking can only be switched off at high effort or below; pairing it with
# xhigh or max is rejected outright. Above that line the planner has to let the
# model think, which is much slower — a deliberate trade, not a default.
EFFORT_REQUIRING_THINKING = ("xhigh", "max")


def read_snapshot(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def write_action_log(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def state_signature(snapshot: dict) -> str:
    """
    Coarse signature used to decide whether the world has changed enough that
    we should re-plan. We intentionally bucket positions so the signature
    changes when a unit actually moves a meaningful distance, not on every
    pixel-jitter — otherwise we'd replan 4x/sec with identical inputs.
    """
    llm_summary = snapshot.get("llm_summary", {})

    def threat_fingerprint(units):
        # Round coordinates to ~8% of the board so we only resignal on
        # meaningful movement, not jitter.
        out = []
        for u in units:
            out.append((
                u.get("label"),
                u.get("lane"),
                round(float(u.get("pressure_score") or 0.0), 1),
            ))
        return sorted(out)

    hand_labels = [s.get("label") for s in snapshot.get("hand_cards_inferred", {}).get("slots", [])]
    payload = {
        "enemy_on_our_side": threat_fingerprint(llm_summary.get("enemy_units_on_friendly_side", [])),
        "friendly_on_enemy_side": threat_fingerprint(llm_summary.get("friendly_units_on_enemy_side", [])),
        "lane_balance": llm_summary.get("lane_balance", {}),
        "hand_labels": hand_labels,
        "elixir": snapshot.get("elixir_inferred", {}).get("blended_estimate"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def verify_action(snapshot: dict, pending_action: dict) -> dict:
    detections = snapshot.get("detections", [])
    card = pending_action["card"]
    target_x = pending_action["capture_target"]["x"]
    target_y = pending_action["capture_target"]["y"]
    sequence = snapshot.get("sequence")

    matches = []
    for det in detections:
        if det.get("owner") != "friendly":
            continue
        label = det.get("label")
        if card == "skeleton" and label == "skeleton":
            pass
        elif label != card:
            continue
        cx = det.get("center", {}).get("x", 0.0)
        cy = det.get("center", {}).get("y", 0.0)
        dist = ((cx - target_x) ** 2 + (cy - target_y) ** 2) ** 0.5
        if dist <= 140:
            matches.append({
                "id": det.get("id"),
                "label": label,
                "distance": round(dist, 2),
                "center": det.get("center"),
                "confidence": det.get("confidence"),
            })

    current_hand = snapshot.get("hand_cards_inferred", {}).get("slots", [])
    same_slot = next((slot for slot in current_hand if slot.get("slot") == pending_action["slot"]), None)
    slot_changed = not same_slot or same_slot.get("label") != card
    if matches:
        return {
            "status": "verified",
            "snapshot_sequence": sequence,
            "reason": "matching friendly unit detected near board tap",
            "matches": matches,
            "slot_changed": slot_changed,
        }

    if sequence - pending_action["snapshot_sequence"] >= 4:
        return {
            "status": "unverified",
            "snapshot_sequence": sequence,
            "reason": "no matching unit detected near board tap within 4 snapshots",
            "slot_changed": slot_changed,
            "current_hand": current_hand,
        }

    return {"status": "pending"}


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Capture Clash Royale state, ask an LLM for a move, and execute it")
    parser.add_argument(
        "--provider", choices=sorted(PLANNERS), default=os.environ.get("LLM_PROVIDER", "openai"),
        help="Which API decides the moves (default: openai, or $LLM_PROVIDER)",
    )
    parser.add_argument(
        "--effort", choices=EFFORT_LEVELS, default=os.environ.get("LLM_EFFORT"),
        help="How hard the planner thinks (or $LLM_EFFORT). Claude defaults to "
             f"{DEFAULT_ANTHROPIC_EFFORT}; OpenAI omits the parameter unless set. "
             "On Claude, xhigh and max force thinking on and are much slower",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model id. Defaults to the provider's default: "
             + ", ".join(f"{name}={model}" for name, model in sorted(DEFAULT_MODELS.items())),
    )
    parser.add_argument("--state-json", default="llm_clasher_state.json")
    parser.add_argument("--decision-json", default="llm_clasher_decision.json")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--cooldown-sec", type=float, default=0.5)
    parser.add_argument(
        "--verify-elixir", action="store_true",
        help="Read the elixir bar either side of every placement to confirm the "
             "game charged for it. Precise, but adds ~0.55s per placement",
    )
    parser.add_argument(
        "--record", metavar="PATH", default=None,
        help="Record the mirrored screen to this mp4 on a background thread",
    )
    parser.add_argument(
        "--record-fps", type=float, default=60.0,
        help="Recording frame rate, independent of the planning rate (default: 60)",
    )
    parser.add_argument(
        "--record-max-size", type=int, default=1392,
        help="Longest edge of the recording in pixels; the window's native size "
             "is 1392, while the detector works at 832 (default: 1392)",
    )
    parser.add_argument(
        "--record-raw", action="store_true",
        help="Record without the detection overlay",
    )
    parser.add_argument(
        "--figure", metavar="PATH", default=None,
        help="Write the latency figure here; defaults to figures/<recording name>.png",
    )
    parser.add_argument(
        "--no-figure", action="store_true",
        help="Skip the latency figure",
    )
    parser.add_argument(
        "--figure-interval-sec", type=float, default=10.0,
        help="How often the figure thread redraws while the match runs (default: 10)",
    )
    parser.add_argument(
        "--stop-file", default=DEFAULT_STOP_FILE,
        help=f"Stop cleanly when this file appears (default: {DEFAULT_STOP_FILE})",
    )
    parser.add_argument(
        "--auto-battle", action="store_true",
        help="Tap through menus and start a new battle whenever one is not running",
    )
    parser.add_argument(
        "--matches", type=int, default=0,
        help="Stop cleanly after this many finished matches (0 = keep playing)",
    )
    parser.add_argument(
        "--idle-exit-sec", type=float, default=90.0,
        help="Without --auto-battle, stop after this long outside a battle (0 = never)",
    )
    parser.add_argument(
        "--battle-load-sec", type=float, default=12.0,
        help="How long to wait after pressing Battle before tapping again",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Skip the press-SPACE gate and start the loop immediately",
    )
    args = parser.parse_args()

    if not args.no_wait:
        wait_for_space()

    # An explicit --model wins; otherwise each provider has its own default, so
    # switching provider does not require also remembering to change the model.
    # OPENAI_MODEL / ANTHROPIC_MODEL still override, matching how the key is read.
    model = args.model or os.environ.get(
        f"{args.provider.upper()}_MODEL", DEFAULT_MODELS[args.provider]
    )
    effort = args.effort or (DEFAULT_ANTHROPIC_EFFORT if args.provider == "anthropic" else None)
    planner_note = f"[INFO] planner: {args.provider} / {model}"
    planner_note += f" / effort {effort}" if effort else " / effort unset (provider default)"
    if args.provider == "anthropic" and effort in EFFORT_REQUIRING_THINKING:
        planner_note += " (thinking on — expect much higher latency)"
    print(planner_note)

    stopper = Stopper(args.stop_file)

    elixir_probe = None
    if args.verify_elixir:
        from .hud_ocr import count_elixir_pips
        from .mirror_capture import MirrorFrameSource

        probe_source = MirrorFrameSource()
        probe_source.probe()

        def elixir_probe():
            frame = probe_source.grab_once()
            if frame is None:
                return None
            return count_elixir_pips(frame).get("count")

        print("[INFO] elixir verification on (adds latency to each placement)")

    recorder = None
    # Shared with the recorder thread: the newest snapshot and decision, so the
    # overlay can be redrawn on every recorded frame. Detection only happens at
    # planning rate, so the boxes refresh at that rate while the video itself
    # stays at the full recording rate.
    latest = {"snapshot": None, "decision": None, "result": None}

    if args.record:
        from .mirror_capture import MirrorFrameSource

        # The recorder owns its own view of the window: the planner's frames
        # arrive through the capture subprocess at planning rate, which is far
        # too slow and too uneven to make a watchable video.
        # Recorded at the window's native size by default: the 832px cap exists
        # for YOLO's benefit and the recorder runs no inference. Detection boxes
        # arrive in detector coordinates and get scaled by the overlay.
        record_source = MirrorFrameSource(max_size=args.record_max_size, target_fps=args.record_fps)
        record_info = record_source.probe()

        def annotate(frame):
            return draw_overlay(frame, latest["snapshot"], latest["decision"], latest["result"])

        recorder = VideoRecorder(
            args.record,
            frame_source=record_source.grab_once,
            fps=args.record_fps,
            annotate=None if args.record_raw else annotate,
        )
        recorder.start()
        print(
            f"[INFO] recording to {args.record} at {args.record_fps:g}fps, "
            f"{record_info['frame']['width']}x{record_info['frame']['height']}"
            f"{' (raw)' if args.record_raw else ' with detection overlay'}"
        )

    # Third thread, alongside gameplay and recording: it only draws, so the loop
    # never pays for it. Named after the recording so a run's video and its
    # latency chart sit under the same stem in two directories.
    figure_recorder = None
    if not args.no_figure:
        from .figures import FigureRecorder, figure_path_for

        figure_path = args.figure or figure_path_for(args.record)
        figure_recorder = FigureRecorder(
            figure_path,
            interval_sec=args.figure_interval_sec,
            title=f"{args.provider} / {model}" + (f", effort {effort}" if effort else ""),
        )
        figure_recorder.start()
        print(f"[INFO] latency figure: {figure_path} (redrawn every {args.figure_interval_sec:g}s)")

    # A state file left by the previous run describes the previous match. Read
    # before the capture writes its first snapshot, it looks like a battle
    # already in progress — enough to report a match starting and then finishing
    # before the real one begins.
    try:
        os.remove(args.state_json)
    except OSError:
        pass

    worker = CaptureWorker(args.python_bin, args.state_json)
    if args.provider == "anthropic":
        planner = AnthropicPlanner(model, effort=args.effort or DEFAULT_ANTHROPIC_EFFORT)
    else:
        planner = OpenAIPlanner(model, effort=args.effort)
    planner_api_errors = planner.transient_api_errors
    executor = MirrorActionExecutor()
    cycle_tracker = CycleTracker()
    tower_tracker = TowerHealthTracker()
    elixir_clock = ElixirClock()
    elixir_clock.start()
    worker.start()

    last_sequence = None
    last_action_ts = 0.0
    last_signature = None
    backoff_until = 0.0
    pending_action = None
    menu_attempt = 0
    next_menu_tap_ts = 0.0
    next_waiting_report_ts = 0.0
    matches_finished = 0
    was_in_battle = False
    left_battle_ts = None
    out_of_battle_streak = 0
    try:
        while True:
            if stopper.should_stop():
                print(f"[INFO] stopping: {stopper.reason}")
                break

            snapshot = read_snapshot(args.state_json)
            if snapshot is None:
                if worker.failed():
                    print(f"[ERROR] clash_capture.py exited with code {worker.exit_code}")
                    return 1
                time.sleep(0.2)
                continue

            sequence = snapshot.get("sequence")
            if sequence is None or sequence == last_sequence:
                time.sleep(0.2)
                continue

            latest["snapshot"] = snapshot

            # Default to "not in a battle" when the capture did not say. The
            # opposite default made a snapshot without the field look like a
            # battle, so the very next one counted as a match finishing.
            in_battle = bool((snapshot.get("screen") or {}).get("in_battle"))

            # Loading screens and the mid-match banners ("GO!", a card name over
            # the arena) hide the elixir bar for a snapshot or two, so a single
            # out-of-battle reading is not the end of a match. Require a short
            # streak before believing it.
            if in_battle:
                out_of_battle_streak = 0
                if not was_in_battle:
                    print("[Match] started")
                    if figure_recorder is not None:
                        figure_recorder.record_event("match start")
                was_in_battle = True
            else:
                out_of_battle_streak += 1
                if was_in_battle and out_of_battle_streak >= MATCH_END_CONFIRM_SNAPSHOTS:
                    matches_finished += 1
                    left_battle_ts = time.time()
                    was_in_battle = False
                    print(f"[Match] finished ({matches_finished})")
                    if figure_recorder is not None:
                        figure_recorder.record_event(f"match end ({matches_finished})")

            if not in_battle:
                if args.matches and matches_finished >= args.matches:
                    print(f"[INFO] stopping: played {matches_finished} match(es)")
                    break

                # Say what the capture is seeing while nothing is happening.
                # Waiting for a battle and failing to recognize the one on
                # screen look identical from outside, and this branch used to
                # be completely silent for however long that lasted.
                now = time.time()
                if now >= next_waiting_report_ts:
                    screen = snapshot.get("screen") or {}
                    print(
                        f"[INFO] waiting: no battle detected (screen={screen.get('kind', 'other')}, "
                        f"snapshot #{sequence})"
                    )
                    next_waiting_report_ts = now + WAITING_REPORT_SEC

                # Never plan against a menu. The LLM has nothing to decide there,
                # and every snapshot would be a paid call to be told so.
                if args.auto_battle:
                    kind = snapshot.get("screen", {}).get("kind", "other")
                    now = time.time()
                    if now >= next_menu_tap_ts:
                        if kind == "home":
                            target = BATTLE_BUTTON
                        else:
                            # Rotate between the two dismissals: the result
                            # banner's OK, and the bottom Battle tab that
                            # returns home from anywhere else.
                            target = OK_BUTTON if menu_attempt % 2 == 0 else BATTLE_TAB
                        print(f"[Menu] screen={kind} tapping {target}")
                        executor.tap_normalized(*target)
                        menu_attempt += 1
                        # A battle takes a few seconds to load; tapping faster
                        # than that just queues taps into the loading screen.
                        next_menu_tap_ts = now + (args.battle_load_sec if kind == "home" else 2.0)
                elif args.idle_exit_sec and left_battle_ts is not None:
                    if time.time() - left_battle_ts >= args.idle_exit_sec:
                        print(f"[INFO] stopping: {args.idle_exit_sec:g}s outside a battle")
                        break

                last_sequence = sequence
                pending_action = None
                continue

            menu_attempt = 0

            if pending_action is not None:
                verification = verify_action(snapshot, pending_action)
                if verification.get("status") in {"verified", "unverified"}:
                    print(f"[AI Action] Verification: {json.dumps(verification)}")
                    pending_action = None

            now = time.time()
            if now - last_action_ts < args.cooldown_sec:
                time.sleep(0.2)
                continue
            if now < backoff_until:
                time.sleep(0.2)
                continue

            tower_tracker.update(snapshot.get("tower_health_inferred", {}).get("towers", {}))
            cycle_tracker.observe_hand(snapshot.get("hand_cards_inferred", {}).get("slots", []))
            snapshot["cycle_state"] = cycle_tracker.state()

            # Soft elixir estimate: blend vision pips with an internal clock so
            # the LLM can reason about elixir even when the pip detector fails.
            observed_elixir = snapshot.get("elixir_inferred", {}).get("count_estimate")
            blended_elixir = elixir_clock.estimate(observed_elixir)
            snapshot.setdefault("elixir_inferred", {})["blended_estimate"] = blended_elixir

            # Inject game phase so the LLM knows about 2x/3x elixir.
            snapshot["_game_phase"] = elixir_clock.game_phase

            last_sequence = sequence
            signature = state_signature(snapshot)
            if signature == last_signature:
                time.sleep(0.2)
                continue

            print(f"[Snapshot] Planning from snapshot #{sequence}")
            try:
                decision = planner.plan(snapshot)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 429:
                    backoff_until = time.time() + 15.0
                    print("[WARN] OpenAI rate limited (429). Backing off for 15s.")
                    continue
                # A rejected request is rejected every time. Retrying it spends
                # the whole match printing the same 400, which is exactly what
                # happened when a model turned out not to accept `temperature`.
                if status is not None and 400 <= status < 500:
                    print(f"[ERROR] OpenAI rejected the request ({status}): {e}")
                    return 1
                # Don't crash the bot on a single transient API error.
                print(f"[WARN] OpenAI HTTP error ({status}): {e}. Idling one tick.")
                backoff_until = time.time() + 2.0
                continue
            except planner_api_errors as e:
                status = getattr(e, "status_code", None)
                # A rejected request is rejected every time — retrying it just
                # burns the match in silence, so say what the API said and stop.
                if status is not None and 400 <= status < 500 and status != 429:
                    print(f"[ERROR] {args.provider} rejected the request ({status}): {e}")
                    return 1
                wait = 15.0 if status == 429 else 2.0
                print(f"[WARN] {args.provider} API error ({status}): {e}. Backing off {wait:g}s.")
                backoff_until = time.time() + wait
                continue
            except (ValueError, json.JSONDecodeError, requests.RequestException) as e:
                print(f"[WARN] Planner error: {e}. Idling one tick.")
                backoff_until = time.time() + 1.0
                continue
            capture_latency_ms = snapshot.get("capture", {}).get("total_latency_ms", 0.0)
            api_latency_ms = planner.last_api_debug.get("latency_ms", 0.0)
            approx_decision_latency_ms = round(capture_latency_ms + api_latency_ms, 2)
            model_decision_latency_sec = planner.estimate_decision_latency_sec(snapshot)
            print(f"[Latency] API: {json.dumps(planner.last_api_debug)}")
            print(
                "[Latency] Summary: "
                + json.dumps({
                    "capture_total_latency_ms": capture_latency_ms,
                    "api_latency_ms": api_latency_ms,
                    "approx_decision_latency_ms": approx_decision_latency_ms,
                    "decision_latency_sec_context": model_decision_latency_sec,
                })
            )
            latest["decision"] = decision
            print(f"[AI Action] Decision: {format_ai_decision(decision)}")

            elixir_before = elixir_probe() if elixir_probe and decision.get("action") == "place_card" else None
            result = executor.execute_decision(decision, snapshot)
            if elixir_before is not None and result.get("status") == "executed":
                time.sleep(ELIXIR_SETTLE_AFTER_PLAY_SEC)
                elixir_after = elixir_probe()
                cost = CARD_COSTS.get(result.get("card"))
                if elixir_after is not None and cost:
                    spent = elixir_before - elixir_after
                    print(
                        f"[Elixir] {result['card']} cost={cost} before={elixir_before} "
                        f"after={elixir_after} spent={spent} "
                        f"charged={'yes' if spent >= cost - ELIXIR_DEBIT_TOLERANCE else 'NO'}"
                    )
            latest["result"] = result
            print(f"[AI Action] Result: {format_ai_result(result)}")
            if figure_recorder is not None:
                figure_recorder.record_decision(
                    capture_latency_ms=capture_latency_ms,
                    api_latency_ms=api_latency_ms,
                    action=decision.get("action", "idle"),
                    card=result.get("card") or decision.get("card"),
                    executed=result.get("status") == "executed",
                    elixir=blended_elixir,
                    usage=planner.last_api_debug.get("usage"),
                )
            # Record action for LLM context (even idles, so the model knows).
            action_record = {
                "action": decision.get("action", "idle"),
                "card": decision.get("card"),
                "t": round(now, 1),
            }
            if decision.get("action") == "idle":
                action_record.pop("card", None)
            planner.recent_actions.append(action_record)

            if result.get("status") == "executed" and result.get("card"):
                cycle_tracker.record_play(result["card"])
                elixir_clock.spend(result["card"])
                pending_action = {
                    "card": result["card"],
                    "slot": result["slot"],
                    "capture_target": result["capture_target"],
                    "snapshot_sequence": sequence,
                }

            write_action_log(
                args.decision_json,
                {
                    "snapshot_sequence": sequence,
                    "cycle_state": snapshot.get("cycle_state", {}),
                    "api_debug": planner.last_api_debug,
                    "capture_latency_ms": capture_latency_ms,
                    "approx_decision_latency_ms": approx_decision_latency_ms,
                    "decision": decision,
                    "execution_result": result,
                    "pending_action": pending_action,
                    "timestamp_unix": now,
                },
            )
            last_action_ts = now
            last_signature = signature
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        worker.stop()
        if figure_recorder is not None:
            figure_path = figure_recorder.stop()
            if figure_path:
                print(f"[INFO] latency figure saved: {figure_path}")
            else:
                # No file is not a drawing failure — it means not one decision
                # completed, which is worth saying rather than leaving a
                # missing file to be discovered later.
                print("[INFO] no latency figure: no decision completed this run")
        if recorder is not None:
            path = recorder.stop()
            if path:
                print(f"[INFO] recording saved: {path} ({recorder.frames_written} frames)")
        # The stop file is deliberately left in place. run_loop.sh runs one of
        # these per match and checks the same file between matches, so consuming
        # it here would swallow the signal and start another match. Whoever
        # starts a run clears it (Stopper does that at construction).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
