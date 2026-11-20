"""Typed reads of stored QSettings values."""

from __future__ import annotations

from PySide6.QtCore import QSettings


def int_setting(settings: QSettings, key: str, default: int) -> int:
    return int(str(settings.value(key, default)))


def bool_setting(settings: QSettings, key: str, default: bool) -> bool:
    return bool(settings.value(key, default, type=bool))
