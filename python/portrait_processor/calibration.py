"""Pick the most detailed accuracy level that fits a seconds-per-photo budget on this PC.

Level 1 is timed for real; the other levels are predicted from their relative cost and the
best candidate is confirmed with one real run, stepping down while it is over budget.
"""

from __future__ import annotations

import json
import os
import platform
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .ai import check_cancel
from .i18n import tr
from .levels import LEVELS, Level, display_name, level

PREFERENCE_KEY = "ai/measurement"


def machine_key() -> str:
    name = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    return f"{name or platform.machine()} · {os.cpu_count() or 1} threads"


@dataclass
class Measurement:
    key: str
    level1: float
    confirmed: dict[int, float] = field(default_factory=dict)
    date: str = ""
    # Seconds of level 1 spent in the background model; the rest (face landmarks, resizing,
    # cleaning) is the same at every level, so only this share scales with the level's cost.
    level1_mask: float | None = None

    def estimate(self, number: int) -> float:
        if number in self.confirmed:
            return self.confirmed[number]
        if number == 1:
            return self.level1
        mask = min(self.level1, self.level1_mask if self.level1_mask is not None else self.level1)
        overhead = self.level1 - mask
        factor = 1.0
        below = [n for n in self.confirmed if n < number]
        if below:
            nearest = max(below)
            factor = max(0.05, (self.confirmed[nearest] - overhead) / (mask * level(nearest).cost))
        return overhead + mask * level(number).cost * factor

    def choice(self, budget: float) -> tuple[Level, float, bool]:
        over = [n for n, seconds in self.confirmed.items() if seconds > budget]
        ceiling = min(over, default=len(LEVELS) + 1)
        candidates = [
            item
            for item in LEVELS
            if item.bundled and item.number < ceiling and self.estimate(item.number) <= budget
        ]
        if not candidates:
            return level(1), self.estimate(1), self.estimate(1) <= budget
        best = candidates[-1]
        return best, self.estimate(best.number), True


def choose(level1_seconds: float, budget: float) -> tuple[Level, float, bool]:
    return Measurement("", level1_seconds).choice(budget)


def _split(timing) -> tuple[float, float | None]:
    """A runner returns total seconds, or (total, seconds spent in the background model)."""
    if isinstance(timing, tuple):
        return float(timing[0]), float(timing[1])
    return float(timing), None


def measure(
    run_level: Callable[[int], float | tuple[float, float]],
    budget: float,
    progress: Callable[[str], None],
    cancel: threading.Event,
) -> Measurement:
    """`run_level(n)` analyses the calibration photo at level n and returns the seconds one
    photo took with the model already loaded."""
    check_cancel(cancel)
    progress(tr("Measuring this PC with {name}…", name=display_name(level(1))))
    run_level(1)  # warm-up: loads the model and fills caches
    check_cancel(cancel)
    total, mask = _split(run_level(1))
    result = Measurement(machine_key(), total, {}, date.today().isoformat(), mask)
    check_cancel(cancel)
    while True:
        chosen, _, _ = result.choice(budget)
        if chosen.number == 1 or chosen.number in result.confirmed:
            return result
        progress(tr("Checking {name} on this PC…", name=display_name(chosen)))
        result.confirmed[chosen.number] = _split(run_level(chosen.number))[0]
        check_cancel(cancel)


def save(preferences, measurement: Measurement) -> None:
    payload = {
        "key": measurement.key,
        "level1": measurement.level1,
        "confirmed": {str(n): s for n, s in measurement.confirmed.items()},
        "date": measurement.date,
        "level1_mask": measurement.level1_mask,
    }
    preferences.setValue(PREFERENCE_KEY, json.dumps(payload))


def load(preferences) -> Measurement | None:
    try:
        payload = json.loads(str(preferences.value(PREFERENCE_KEY, "")))
        known = {item.number for item in LEVELS}
        mask = payload.get("level1_mask")
        result = Measurement(
            str(payload["key"]),
            float(payload["level1"]),
            {int(n): float(s) for n, s in payload["confirmed"].items() if int(n) in known},
            str(payload["date"]),
            None if mask is None else float(mask),
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        # A damaged or outdated measurement just means measuring again.
        return None
    return result if result.key == machine_key() else None
