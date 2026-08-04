#!/usr/bin/env python3
"""
Screen recognition and menu navigation between matches.

The planner only makes sense inside a battle. Everything else — the home
screen, the post-match result banner, a chest or reward popup — is menu
plumbing that has to be tapped through before the next battle starts. This
module classifies which of those screens is showing and taps accordingly, so
the loop can run unattended across many matches.

Classification is colour based rather than OCR based, for the same reason the
elixir count is: the game's stylized font is unreliable to read, while these
screens differ enormously in colour.
"""

from typing import Optional

import cv2
import numpy as np

from .capture_config import (
    ELIXIR_PIP_X_LEFT_NORM,
    ELIXIR_PIP_X_RIGHT_NORM,
    ELIXIR_PIP_Y_LOWER_NORM,
    ELIXIR_PIP_Y_UPPER_NORM,
)


# The elixir bar is the giveaway that a battle is running: every segment is
# either filled pink or empty navy, so the two together cover nearly the whole
# band. Menus put arena art or a dialog there instead and score far lower.
# Measured: battles 0.91-0.93, home screen 0.41, result banner 0.07.
IN_BATTLE_MIN_BAR_COVERAGE = 0.80

# The home screen's yellow Battle button.
BATTLE_BUTTON = (0.505, 0.798)
BATTLE_BUTTON_PROBE = (0.44, 0.56, 0.78, 0.82)
HOME_MIN_YELLOW = 0.40

# The result banner's OK button. Reward and chest popups also dismiss from
# roughly here, or from a tap on dead space.
OK_BUTTON = (0.497, 0.826)

# Bottom navigation "Battle" tab, used to get back to the home screen when a
# popup leaves the game on some other tab.
BATTLE_TAB = (0.58, 0.92)


def _crop(frame: np.ndarray, x1: float, x2: float, y1: float, y2: float) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def elixir_bar_coverage(frame: np.ndarray) -> float:
    """Fraction of the elixir bar band that is either filled or empty segment."""
    band = _crop(
        frame,
        ELIXIR_PIP_X_LEFT_NORM,
        ELIXIR_PIP_X_RIGHT_NORM,
        ELIXIR_PIP_Y_LOWER_NORM,
        ELIXIR_PIP_Y_UPPER_NORM,
    )
    if band.size == 0:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    pink = (hue >= 135) & (hue <= 175) & (sat >= 90) & (val >= 90)
    navy = (hue >= 100) & (hue <= 130) & (sat >= 90) & (val >= 40) & (val <= 170)
    return float((pink | navy).mean())


def battle_button_yellow(frame: np.ndarray) -> float:
    patch = _crop(frame, *BATTLE_BUTTON_PROBE)
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 120)).mean())


def is_in_battle(frame: np.ndarray) -> bool:
    return elixir_bar_coverage(frame) >= IN_BATTLE_MIN_BAR_COVERAGE


def screen_kind(frame: np.ndarray) -> str:
    """One of 'battle', 'home', 'other'."""
    if is_in_battle(frame):
        return "battle"
    if battle_button_yellow(frame) >= HOME_MIN_YELLOW:
        return "home"
    return "other"


def next_tap(frame: np.ndarray, attempt: int = 0) -> Optional[tuple]:
    """Where to tap to get closer to a battle, or None if already in one.

    On an unrecognized screen the taps rotate: first the OK button, since the
    post-match banner is by far the most common case, then the bottom Battle
    tab, which returns to the home screen from anywhere else.
    """
    kind = screen_kind(frame)
    if kind == "battle":
        return None
    if kind == "home":
        return BATTLE_BUTTON
    return OK_BUTTON if attempt % 2 == 0 else BATTLE_TAB


def main() -> int:
    """Diagnostic: report what screen the mirror window is showing."""
    from .mirror_capture import MirrorFrameSource

    source = MirrorFrameSource()
    source.probe()
    frame = source.grab_once()
    if frame is None:
        print("[ERROR] no frame")
        return 1
    print(
        f"[INFO] screen={screen_kind(frame)} "
        f"bar_coverage={elixir_bar_coverage(frame):.3f} "
        f"battle_button_yellow={battle_button_yellow(frame):.3f} "
        f"next_tap={next_tap(frame)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
