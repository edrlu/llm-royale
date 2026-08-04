#!/usr/bin/env python3
"""
Device actions for Clash Royale.

The concrete backend lives in `mac_action.MirrorActionExecutor`, which drives a
real iPhone through the macOS "iPhone Mirroring" window. This module holds the
device-agnostic half: normalized coordinates, hand slot geometry, and turning a
planner decision into taps and drags.
"""

import sys
import time
from typing import Dict, Optional


# Tap targets for the four hand slots, measured on the iPhone Mirroring window.
# These must stay in sync with HAND_SLOT_CENTERS in capture_config: that module
# reads a slot's card art at these columns, and this one taps it.
CARD_SLOT_X_NORM = {
    1: 0.317,
    2: 0.490,
    3: 0.663,
    4: 0.836,
}
CARD_SLOT_Y_NORM = 0.880


class BaseActionExecutor:
    """Device-agnostic half of the action layer.

    Subclasses supply three primitives in device coordinate space —
    `ensure_device_ready`, `tap_device`, `swipe_device` — plus `device_width`
    and `device_height`. Everything above that (normalized coords, hand slot
    selection, decision execution) is backend independent.
    """

    device_width = None
    device_height = None
    device_serial = None

    def ensure_device_ready(self) -> None:
        raise NotImplementedError

    def tap_device(self, x: int, y: int) -> Dict:
        raise NotImplementedError

    def swipe_device(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 120) -> Dict:
        raise NotImplementedError

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
                "reason": "swipe command failed",
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
