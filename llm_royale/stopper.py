#!/usr/bin/env python3
"""
Kill switch for a running bot.

The bot drives a real phone playing real ladder matches, so there has to be a
way to stop it that does not depend on having the terminal it was started from
in front of you. Three routes, all landing on the same flag:

- a stop file appearing on disk (`touch STOP` from anywhere, or the
  `python -m llm_royale.stopper` helper)
- SIGINT or SIGTERM, so `kill` and Ctrl-C shut down cleanly instead of killing
  the process mid-frame and losing the recording
- `request_stop()` from inside the process

Shutdown is cooperative: the loop notices the flag on its next pass and unwinds
normally, which is what lets the recorder finalize its video and the capture
subprocess exit.
"""

import os
import signal
import threading
import time


DEFAULT_STOP_FILE = "STOP"


class Stopper:
    def __init__(self, stop_file: str = DEFAULT_STOP_FILE, install_signal_handlers: bool = True):
        self.stop_file = stop_file
        self._event = threading.Event()
        self._reason = None

        # A stop file left over from a previous run would stop this one before
        # it played a single card.
        self.clear_stop_file()

        if install_signal_handlers:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(sig, self._on_signal)
                except (ValueError, OSError):
                    # Not on the main thread; the file check still works.
                    pass

    def _on_signal(self, signum, _frame) -> None:
        self.request_stop(f"signal {signal.Signals(signum).name}")

    def request_stop(self, reason: str = "requested") -> None:
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    @property
    def reason(self) -> str:
        return self._reason or "unknown"

    def clear_stop_file(self) -> None:
        try:
            os.remove(self.stop_file)
        except OSError:
            pass

    def should_stop(self) -> bool:
        if self._event.is_set():
            return True
        if self.stop_file and os.path.exists(self.stop_file):
            self.request_stop(f"stop file {self.stop_file}")
            return True
        return False


def main() -> int:
    """Ask a running bot to stop by creating its stop file."""
    import argparse

    parser = argparse.ArgumentParser(description="Stop a running llm-royale bot")
    parser.add_argument("--stop-file", default=DEFAULT_STOP_FILE)
    parser.add_argument(
        "--wait", type=float, default=15.0,
        help="Seconds to wait for the bot to consume the stop file",
    )
    args = parser.parse_args()

    with open(args.stop_file, "w", encoding="utf-8") as handle:
        handle.write(f"stop requested at {time.time():.0f}\n")
    print(f"[INFO] wrote {args.stop_file}")

    deadline = time.time() + args.wait
    while time.time() < deadline:
        if not os.path.exists(args.stop_file):
            print("[INFO] bot acknowledged the stop and is shutting down")
            return 0
        time.sleep(0.25)
    print("[WARN] stop file still present — the bot may not be running")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
