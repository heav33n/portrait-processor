"""Background-removal accuracy levels, from fastest to most detailed."""

from __future__ import annotations

from dataclasses import dataclass

from .i18n import tr


@dataclass(frozen=True)
class Level:
    number: int
    name: str
    model: str
    refine: bool
    bundled: bool
    # Seconds per photo relative to level 1, measured with two threads (tools/pack_models.py,
    # Ryzen 5625U: 0.65 / +0.08 refine / 1.84 / 13.3 / 39.3 s). Auto confirms on the real PC.
    cost: float


LEVELS = (
    Level(1, "Fastest", "u2net_human_seg", False, True, 1.0),
    Level(2, "Fast · refined edges", "u2net_human_seg", True, True, 1.15),
    Level(3, "Balanced", "isnet-general-use", False, True, 2.8),
    Level(4, "Detailed", "birefnet-general-lite", False, True, 20.0),
    Level(5, "Maximum", "birefnet-portrait", False, False, 60.0),
)
LEGACY_MODELS = {"u2net_human_seg": 1, "birefnet-portrait": 5}


def display_name(item: Level) -> str:
    """The level's name in the interface language; Level.name stays the English source."""
    names = {
        1: tr("Fastest"),
        2: tr("Fast · refined edges"),
        3: tr("Balanced"),
        4: tr("Detailed"),
        5: tr("Maximum"),
    }
    return names.get(item.number, item.name)


def level(number: int) -> Level:
    for item in LEVELS:
        if item.number == number:
            return item
    raise ValueError(tr("Unknown accuracy level: {number}", number=number))


def level_from_preference(value) -> int | None:
    """Map a stored preference to a level number; None means Auto."""
    text = str(value).strip()
    if text in LEGACY_MODELS:
        return LEGACY_MODELS[text]
    if text.isdigit() and any(item.number == int(text) for item in LEVELS):
        return int(text)
    return None
