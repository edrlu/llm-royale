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
    reason = str(decision.get("reason", "")).strip()

    if action == "place_card":
        parts = [f"play {decision.get('card', '?')}"]
        x_norm = decision.get("x_norm")
        y_norm = decision.get("y_norm")
        if x_norm is not None and y_norm is not None:
            parts.append(f"at ({float(x_norm):.2f}, {float(y_norm):.2f})")
        elif decision.get("x") is not None and decision.get("y") is not None:
            parts.append(f"at ({float(decision['x']):.0f}, {float(decision['y']):.0f})")
        if reason:
            parts.append(f"because {reason}")
        return " ".join(parts)

    if reason:
        return f"{action}: {reason}"
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


class OpenAIPlanner:
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.last_api_debug = {}
        self.recent_api_latencies_ms = deque(maxlen=8)
        self.recent_actions: deque = deque(maxlen=4)

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

    def build_instructions(self, width: int, height: int, decision_latency_sec: float, river_y: float = 0.5) -> str:
        R = river_y
        return (
            "You are a top-ladder Hog 2.6 player. Deck: hog-rider(4), musketeer(4), "
            "ice-spirit(1), ice-golem(2), fireball(4), the-log(2), cannon(3), skeleton(1).\n"
            "Return STRICT JSON ONLY, one of:\n"
            '{"action":"idle","reason":"..."}\n'
            '{"action":"place_card","card":"<name>","x_norm":<0..1>,"y_norm":<0..1>,"reason":"..."}\n'
            "\n"
            "COORDINATE SYSTEM:\n"
            f"Normalized 0-1. Top of screen = 0.0 (enemy king). River/bridge ≈ {R:.2f}.\n"
            f"Your side = y > {R:.2f}. Enemy side = y < {R:.2f}. Arena ends ~0.80 (below is hand bar).\n"
            "Left lane x≈0.22, right lane x≈0.78. Center x≈0.50.\n"
            "\n"
            # Measured from detected tower boxes on the iPhone capture. The lanes
            # sit further out than they look: 0.22/0.78, not 0.33/0.67.
            "LANDMARK POSITIONS (approximate y_norm):\n"
            "- Enemy king tower:      y≈0.21, x≈0.50\n"
            "- Enemy princess towers:  y≈0.29, left x≈0.22, right x≈0.78\n"
            f"- Bridge/river:           y≈{R:.2f}, bridges at x≈0.22 and x≈0.78\n"
            "- YOUR princess towers:   y≈0.64, left x≈0.22, right x≈0.78\n"
            "- YOUR king tower:        y≈0.72, x≈0.50\n"
            "The `towers` dict has LIVE detected positions (x,y) — use those when available, "
            "fall back to these landmarks only if a tower is missing from detection.\n"
            "\n"
            "HOG 2.6 PRO STRATEGY:\n"
            "\n"
            "DEFENSE:\n"
            "- Cannon: place it between your two princess towers (x≈0.50, y between your princess towers "
            "and river). A 4-3 plant (4 tiles from river, 3 tiles from center) pulls hog to cannon "
            "with both princess towers in range. Check `towers` for your princess tower y-coords.\n"
            "- Musketeer deep behind the cannon — she outranges most troops and stays alive to counter-push.\n"
            "- Ice-golem kites melee units to the opposite lane. Drop it where you want the enemy to walk.\n"
            "- Skeletons surround high-DPS single-target units (mini pekka, prince, pekka). Place on top of them.\n"
            "- Ice-spirit freezes and resets. Great vs inferno tower, sparky, prince charge.\n"
            "- Log clears swarms (skeleton army, goblin barrel, princess). Roll it through them.\n"
            "\n"
            "OFFENSE — Hog 2.6 wins by OUTCYCLING:\n"
            "- Hog at the bridge (y≈river_y from `board`) in the lane the opponent just defended. They can't defend both.\n"
            "- After defending, IMMEDIATELY counter-push hog opposite lane. This is the #1 win condition.\n"
            "- Ice-spirit or ice-golem in front of hog tanks one hit and lets hog get extra swings.\n"
            "- Fireball + log chip on a low-HP princess tower to finish it off.\n"
            "- Fireball value: hit enemy troops clustered near their tower to get tower chip + troop damage.\n"
            "- In 2x/3x elixir be much more aggressive — cycle hog faster, pressure both lanes.\n"
            "\n"
            "SPELL USAGE:\n"
            "- Log: aim at the enemy tower's x,y from `towers` to chip it, or at enemy troops' x_norm/y_norm. "
            "Log rolls DOWNWARD from where you place it — place it slightly ABOVE (lower y_norm) your target.\n"
            "- Fireball: aim at the enemy unit's x_norm/y_norm position, or at the tower's x,y for chip. "
            "Fireball hits a circle around the target point.\n"
            "- NEVER throw spells at empty space. Always target something visible in the snapshot.\n"
            "\n"
            "DATA FIELDS:\n"
            "- `enemy_on_our_side`: enemies that crossed the river — DEFEND THESE FIRST.\n"
            "- `friendly_on_enemy_side` / `friendly_troops`: your surviving units.\n"
            "- `towers`: each tower has `hp` (0-100%), `x` and `y` (detected position). "
            "Use tower x/y to aim spells AT the tower. Use tower positions to place cannon between your princess towers.\n"
            "- `game_phase`: 1x/2x/3x elixir speed. Push more aggressively in 2x/3x.\n"
            "- `cycle`: recent plays + upcoming cards. Plan your next combo.\n"
            "- `recent_actions`: what you just did — don't repeat the same card back-to-back.\n"
            "- Unit positions have x_norm/y_norm — use these coords when targeting spells on them.\n"
            "\n"
            "PRIORITIES:\n"
            "1. Defend enemy pushes — always. Use the cheapest card that works.\n"
            "2. Counter-push with hog after defending. Opposite lane.\n"
            "3. Support existing pushes with cheap cards behind your hog.\n"
            "4. Start hog push when you have elixir advantage and no threat.\n"
            "5. Spell-chip a low tower if fireball/log can finish it.\n"
            "6. At 9+ elixir, play something cheap in the back to avoid leaking.\n"
            "7. Otherwise IDLE — patience wins. Don't overcommit.\n"
            "\n"
            f"CONTROL LAG: ~{decision_latency_sec:.1f}s. Lead moving targets slightly.\n"
            "\n"
            "RULES:\n"
            "- ONLY play cards in `playable_cards`. If empty → IDLE.\n"
            "- null hand slot = hidden card, not empty.\n"
            "- Never play a card you can't afford (check `elixir` vs `playable_with_cost`).\n"
            "- Keep `reason` <= 8 words."
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
                    "reason": {"type": "string"},
                },
                "required": ["action", "card", "x_norm", "y_norm", "reason"],
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

    def _extract_decision(self, data: dict) -> dict:
        decision = json.loads(self._extract_output_text(data))
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
        compact = self.build_compact_snapshot(snapshot)
        width = compact["w"]
        height = compact["h"]
        decision_latency_sec = compact["decision_latency_sec"]
        river_y = float(compact["board"].get("river_y_norm") or 0.5)
        data = self._responses_create(
            instructions=self.build_instructions(width, height, decision_latency_sec, river_y),
            input_text=self.build_input(snapshot),
        )
        return self._extract_decision(data)


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
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--state-json", default="llm_clasher_state.json")
    parser.add_argument("--decision-json", default="llm_clasher_decision.json")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--cooldown-sec", type=float, default=0.5)
    parser.add_argument(
        "--record", metavar="PATH", default=None,
        help="Record the mirrored screen to this mp4 on a background thread",
    )
    parser.add_argument(
        "--record-fps", type=float, default=12.0,
        help="Recording frame rate, independent of the planning rate (default: 12)",
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

    stopper = Stopper(args.stop_file)

    recorder = None
    if args.record:
        from .mirror_capture import MirrorFrameSource

        # The recorder owns its own view of the window: the planner's frames
        # arrive through the capture subprocess at planning rate, which is far
        # too slow and too uneven to make a watchable video.
        record_source = MirrorFrameSource(target_fps=args.record_fps)
        record_source.probe()
        recorder = VideoRecorder(
            args.record,
            frame_source=record_source.grab_once,
            fps=args.record_fps,
        )
        recorder.start()
        print(f"[INFO] recording to {args.record} at {args.record_fps:g}fps")

    worker = CaptureWorker(args.python_bin, args.state_json)
    planner = OpenAIPlanner(args.model)
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
    matches_finished = 0
    was_in_battle = False
    left_battle_ts = None
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

            in_battle = snapshot.get("screen", {}).get("in_battle", True)

            if was_in_battle and not in_battle:
                matches_finished += 1
                left_battle_ts = time.time()
                print(f"[Match] finished ({matches_finished})")
            was_in_battle = in_battle

            if not in_battle:
                if args.matches and matches_finished >= args.matches:
                    print(f"[INFO] stopping: played {matches_finished} match(es)")
                    break

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
                # Don't crash the bot on a single transient API error.
                print(f"[WARN] OpenAI HTTP error ({status}): {e}. Idling one tick.")
                backoff_until = time.time() + 2.0
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
            print(f"[AI Action] Decision: {format_ai_decision(decision)}")
            result = executor.execute_decision(decision, snapshot)
            print(f"[AI Action] Result: {format_ai_result(result)}")
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
        if recorder is not None:
            path = recorder.stop()
            if path:
                print(f"[INFO] recording saved: {path} ({recorder.frames_written} frames)")
        stopper.clear_stop_file()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
