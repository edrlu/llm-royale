#!/usr/bin/env python3
"""
Android device actions for Clash Royale.
"""

import shutil
import subprocess
import sys
import time
from typing import Dict, Optional


CARD_SLOT_X_NORM = {
    1: 0.35,
    2: 0.50,
    3: 0.68,
    4: 0.83,
}
CARD_SLOT_Y_NORM = 0.91


def find_adb() -> str:
    adb = shutil.which("adb")
    if adb:
        return adb
    for candidate in ("/opt/homebrew/bin/adb", "/usr/local/bin/adb"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("adb not found")


def list_connected_devices(adb_bin: str) -> list[str]:
    out = subprocess.check_output(
        [adb_bin, "devices"],
        stderr=subprocess.DEVNULL,
        timeout=5,
    ).decode(errors="replace")
    devices = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def get_device_resolution(adb_bin: str) -> tuple[int, int]:
    out = subprocess.check_output(
        [adb_bin, "shell", "wm", "size"],
        stderr=subprocess.DEVNULL,
        timeout=5,
    ).decode().strip()
    last_line = out.splitlines()[-1]
    width_s, height_s = last_line.split(":")[-1].strip().split("x")
    return int(width_s), int(height_s)


class AndroidActionExecutor:
    def __init__(self):
        self.adb_bin = find_adb()
        self.device_width = None
        self.device_height = None
        self.device_serial = None

    def ensure_device_ready(self) -> None:
        subprocess.run([self.adb_bin, "start-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        devices = list_connected_devices(self.adb_bin)
        if not devices:
            raise RuntimeError(
                "No Android device is connected through adb. Check USB/debugging and run `adb devices`."
            )
        self.device_serial = devices[0]
        if self.device_width is None or self.device_height is None:
            try:
                self.device_width, self.device_height = get_device_resolution(self.adb_bin)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "adb found a device, but `adb shell wm size` failed. Unlock the phone and confirm USB debugging."
                ) from e

    def tap_device(self, x: int, y: int) -> Dict:
        self.ensure_device_ready()
        cmd = [self.adb_bin, "shell", "input", "tap", str(x), str(y)]
        started = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration_ms = (time.time() - started) * 1000.0
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "duration_ms": round(duration_ms, 2),
            "x": x,
            "y": y,
        }

    def swipe_device(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 120) -> Dict:
        self.ensure_device_ready()
        cmd = [
            self.adb_bin, "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms),
        ]
        started = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_ms = (time.time() - started) * 1000.0
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "duration_ms": round(elapsed_ms, 2),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "swipe_duration_ms": duration_ms,
        }

    def tap_normalized(self, x_norm: float, y_norm: float) -> Dict:
        self.ensure_device_ready()
        x = max(0, min(self.device_width - 1, int(round(x_norm * self.device_width))))
        y = max(0, min(self.device_height - 1, int(round(y_norm * self.device_height))))
        result = self.tap_device(x, y)
        result["x_norm"] = round(x_norm, 4)
        result["y_norm"] = round(y_norm, 4)
        return result

    def select_hand_slot(self, slot: int) -> Dict:
        if slot not in CARD_SLOT_X_NORM:
            raise ValueError(f"invalid hand slot: {slot}")
        result = self.tap_normalized(CARD_SLOT_X_NORM[slot], CARD_SLOT_Y_NORM)
        result["slot"] = slot
        return result

    def hand_slot_to_device(self, slot: int) -> Dict:
        self.ensure_device_ready()
        if slot not in CARD_SLOT_X_NORM:
            raise ValueError(f"invalid hand slot: {slot}")
        x_norm = CARD_SLOT_X_NORM[slot]
        y_norm = CARD_SLOT_Y_NORM
        x = max(0, min(self.device_width - 1, int(round(x_norm * self.device_width))))
        y = max(0, min(self.device_height - 1, int(round(y_norm * self.device_height))))
        return {"slot": slot, "x": x, "y": y, "x_norm": round(x_norm, 4), "y_norm": round(y_norm, 4)}

    def capture_to_device(self, x: float, y: float, capture_width: int, capture_height: int) -> tuple[int, int]:
        self.ensure_device_ready()
        x_norm = x / capture_width
        y_norm = y / capture_height
        device_x = max(0, min(self.device_width - 1, int(round(x_norm * self.device_width))))
        device_y = max(0, min(self.device_height - 1, int(round(y_norm * self.device_height))))
        return device_x, device_y

    def execute_decision(self, decision: dict, snapshot: dict) -> dict:
        action = str(decision.get("action", "idle")).lower()
        if action == "idle":
            return {"status": "idle", "reason": decision.get("reason", "")}

        if action != "place_card":
            return {"status": "rejected", "reason": f"unsupported action {action}"}

        try:
            self.ensure_device_ready()
        except RuntimeError as e:
            return {"status": "rejected", "reason": str(e)}

        card = decision.get("card")
        if not card:
            return {"status": "rejected", "reason": "missing card"}

        capture = snapshot["capture"]
        capture_width = int(capture["width"])
        capture_height = int(capture["height"])

        if "x_norm" in decision and "y_norm" in decision:
            x = float(decision["x_norm"]) * capture_width
            y = float(decision["y_norm"]) * capture_height
        else:
            x = float(decision.get("x"))
            y = float(decision.get("y"))

        x_norm = x / capture_width
        y_norm = y / capture_height

        # Only hard clamp: keep coords inside the capture frame.
        x_norm = min(max(x_norm, 0.01), 0.99)
        y_norm = min(max(y_norm, 0.01), 0.99)
        x = x_norm * capture_width
        y = y_norm * capture_height

        hand_slots = snapshot.get("hand_cards_inferred", {}).get("slots", [])
        selected_slot = next((slot for slot in hand_slots if slot.get("label") == card), None)

        # Only one safe fallback: the card isn't claimed by any slot but there's
        # EXACTLY one unlabeled slot, so by elimination that's where it is.
        # Any other "fallback" (lowest-confidence guess, cycle-tracker memory)
        # risks playing the wrong card — e.g. the LLM asks for the-log and we
        # accidentally swipe a 4-elixir ice-golem to the bridge.
        if selected_slot is None:
            empty_slots = [s for s in hand_slots if not s.get("label")]
            labeled = [s.get("label") for s in hand_slots if s.get("label")]
            # Only accept this branch if the card is missing from every labeled
            # slot (otherwise labeled slots are authoritative) and there is
            # exactly one unlabeled slot to fill.
            if len(empty_slots) == 1 and card not in labeled:
                selected_slot = empty_slots[0]

        if selected_slot is None:
            return {
                "status": "rejected",
                "reason": f"{card} is not currently inferred in hand",
                "debug": {
                    "available_hand_slots": hand_slots,
                },
            }

        slot_num = int(selected_slot["slot"])
        slot_anchor = self.hand_slot_to_device(slot_num)
        target_x, target_y = self.capture_to_device(x, y, capture_width, capture_height)
        swipe_result = self.swipe_device(slot_anchor["x"], slot_anchor["y"], target_x, target_y, duration_ms=110)
        if swipe_result["returncode"] != 0:
            return {
                "status": "failed",
                "reason": "adb swipe command failed",
                "debug": {
                    "device_serial": self.device_serial,
                    "device_resolution": {"width": self.device_width, "height": self.device_height},
                    "slot_anchor": slot_anchor,
                    "swipe": swipe_result,
                    "hand_slots": hand_slots,
                },
            }

        return {
            "status": "executed",
            "card": card,
            "slot": slot_num,
            "selection_tap": {
                "x": slot_anchor["x"],
                "y": slot_anchor["y"],
                "x_norm": slot_anchor["x_norm"],
                "y_norm": slot_anchor["y_norm"],
            },
            "board_tap": {"x": target_x, "y": target_y},
            "capture_target": {
                "x": round(x, 2),
                "y": round(y, 2),
                "x_norm": round(x_norm, 4),
                "y_norm": round(y_norm, 4),
            },
            "device_serial": self.device_serial,
            "device_resolution": {"width": self.device_width, "height": self.device_height},
            "debug": {
                "selected_slot_inference": selected_slot,
                "hand_slots": hand_slots,
                "slot_anchor": slot_anchor,
                "swipe": swipe_result,
            },
            "reason": decision.get("reason", ""),
        }


def main() -> int:
    print("This module is intended to be imported by llm_clasher.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
