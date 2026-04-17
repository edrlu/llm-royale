#!/usr/bin/env python3
"""
Run Clash capture, call an LLM for the next move, and execute the action on Android.
"""

import argparse
import json
import os
import select
import re
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from typing import Optional

import requests

from .action import AndroidActionExecutor
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

        # --- Tower health as percentage (0-100) ---
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
                towers[key] = pct

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

    def build_instructions(self, width: int, height: int, decision_latency_sec: float) -> str:
        return (
            "You play Hog 2.6. Deck: hog-rider(4), musketeer(4), ice-spirit(1), ice-golem(2), "
            "fireball(4), the-log(2), cannon(3), skeleton(1).\n"
            "Return STRICT JSON ONLY, one of:\n"
            '{"action":"idle","reason":"..."}\n'
            '{"action":"place_card","card":"<name>","x_norm":<0..1>,"y_norm":<0..1>,"reason":"..."}\n'
            "\n"
            f"CAPTURE FRAME: {width}x{height}px. Arena = top ~78%; bottom 22% = hand/elixir (OFF-LIMITS).\n"
            "Use normalized coords from `board.place_bbox_norm` and `board.river_y_norm`.\n"
            "\n"
            "KEY Y-COORDS:\n"
            "  enemy king ~= river-0.33, enemy princess ~= river-0.20, bridge = river,\n"
            "  friendly princess ~= river+0.20, friendly king ~= river+0.30, back ~= river+0.35.\n"
            "X LANES: left ~0.30, center ~0.50, right ~0.70.\n"
            "\n"
            "PLACEMENT RULES:\n"
            "- Troops/cannon: y_norm >= river_y_norm, inside `board.place_bbox_norm`.\n"
            "- fireball & the-log: anywhere in arena (y_norm 0.0-0.78).\n"
            "- x_norm must be in [0.05, 0.95].\n"
            "- NEVER spell within 0.12 of a friendly unit. Never spell empty space.\n"
            "\n"
            "DEFENSIVE DEFAULTS:\n"
            "- cannon: x=0.50, y=river+0.18 (pulls Hog/Ram/RG to center).\n"
            "- musketeer: x=0.50, y=river+0.25 (deep, safe behind cannon).\n"
            "- ice-golem kite: x=opposite lane, y=river+0.15.\n"
            "- skeletons/ice-spirit: x=threat lane, y=river+0.10.\n"
            "\n"
            "OFFENSIVE DEFAULTS:\n"
            "- hog solo: x=0.30 or 0.70, y=river+0.02 (at bridge).\n"
            "- the-log chip: x=tower lane, y=river-0.12.\n"
            "- fireball chip: onto enemy princess tower (only if tower HP is low enough that 2-3 fireballs kill it).\n"
            "\n"
            "CONTEXT FIELDS:\n"
            "- `towers`: e.g. {\"enemy_left\":85,\"friendly_king\":100} = HP percentage (0=dead, 100=full).\n"
            "- `game_phase`: \"1x\", \"2x\", or \"3x\" elixir regen speed.\n"
            "- `cycle`: last cards played + upcoming cards.\n"
            "- `recent_actions`: what you already did in the last few seconds — do NOT repeat.\n"
            "\n"
            "DECISION LOGIC (in priority order):\n"
            "1. DEFEND: If `enemy_on_our_side` non-empty → play cheapest sufficient counter, "
            "even at low elixir. Hog/Ram/RG → cannon pull. Swarms → log/ice-spirit. Tanks → cannon+musk.\n"
            "2. COUNTER-PUSH: After defending with surviving troops on the board, immediately hog the "
            "OPPOSITE lane while the opponent is low on elixir. This is Hog 2.6's core win condition.\n"
            "3. SUPPORT PUSH: If `friendly_on_enemy_side` exists AND elixir >= 5 → support with ice-spirit/ice-golem/musketeer.\n"
            "4. START PUSH: If elixir >= 7 AND no threat → hog at bridge, opposite last enemy investment.\n"
            "   In 2x/3x elixir, push at elixir >= 6. At elixir >= 9, you MUST play something (leaking is always wrong).\n"
            "5. CHIP: If enemy tower in `towers` <= 20% and fireball/log in hand → spell to finish it.\n"
            "6. IDLE: Otherwise wait.\n"
            "\n"
            f"CONTROL LAG: ~{decision_latency_sec:.1f}s from snapshot to tap. Lead moving targets.\n"
            "\n"
            "HARD RULES:\n"
            "- ONLY play cards in `playable_cards`. If empty → IDLE.\n"
            "- null hand slot = unaffordable hidden card, not empty.\n"
            "- Never play a card costing more than current `elixir`.\n"
            "- Check `recent_actions` — do NOT double-play the same card within 3s unless defending.\n"
            "- Keep `reason` <= 8 words."
        )

    def build_input(self, snapshot: dict) -> str:
        compact = self.build_compact_snapshot(snapshot)
        return json.dumps(compact, separators=(",", ":"))

    def _extract_output_text(self, data: dict) -> str:
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()
        chunks = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks).strip()

    def _extract_json_text(self, data: dict) -> str:
        text = self._extract_output_text(data)
        if not text:
            raise ValueError("Empty model response")
        # Strip code fences the model sometimes wraps JSON in.
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass
        # Greedy but non-greedy per line; take the largest balanced-looking object.
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model did not return parseable JSON: {text[:300]}")
        candidate = match.group(0)
        json.loads(candidate)
        return candidate

    def _responses_create(self, *, instructions: str, input_text: str) -> dict:
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "temperature": 0,
            # 60 was so tight the model could truncate mid-JSON on any reasoning;
            # 220 gives headroom while still capping cost. The JSON itself is
            # ~40-60 tokens; the rest is headroom for hidden reasoning tokens.
            "max_output_tokens": 220,
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
        data = self._responses_create(
            instructions=self.build_instructions(width, height, decision_latency_sec),
            input_text=self.build_input(snapshot),
        )
        return json.loads(self._extract_json_text(data))


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
    args = parser.parse_args()

    wait_for_space()

    worker = CaptureWorker(args.python_bin, args.state_json)
    planner = OpenAIPlanner(args.model)
    executor = AndroidActionExecutor()
    cycle_tracker = CycleTracker()
    elixir_clock = ElixirClock()
    elixir_clock.start()
    worker.start()

    last_sequence = None
    last_action_ts = 0.0
    last_signature = None
    backoff_until = 0.0
    pending_action = None
    try:
        while True:
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
            print(f"[AI Action] Decision: {json.dumps(decision)}")
            result = executor.execute_decision(decision, snapshot)
            print(f"[AI Action] Result: {json.dumps(result)}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
