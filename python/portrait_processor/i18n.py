"""Interface language: English source text, translated through a per-language catalog.

`tr("Saved {name}.", name=n)` looks up the English text and formats the result. The language
follows the system unless the user chose one; it is fixed for the life of the process, and
child processes inherit it through PORTRAIT_LANGUAGE.
"""

from __future__ import annotations

import os

LANGUAGES = {"en": "English", "es": "Español"}
ENVIRONMENT_KEY = "PORTRAIT_LANGUAGE"
_language: str | None = None


def _catalog(code: str) -> dict[str, str]:
    if code == "es":
        from .i18n_es import ES

        return ES
    return {}


def system_language() -> str:
    try:
        from PySide6.QtCore import QLocale

        name = QLocale.system().name()
    except ImportError:
        name = os.environ.get("LANG", "")
    return "es" if name.lower().startswith("es") else "en"


def resolve(preference) -> str:
    """'en' or 'es' for an explicit choice; anything else ('auto', None) follows the system."""
    text = str(preference or "").strip().lower()
    return text if text in LANGUAGES else system_language()


def set_language(code: str) -> None:
    global _language
    _language = code if code in LANGUAGES else "en"
    os.environ[ENVIRONMENT_KEY] = _language


def language() -> str:
    global _language
    if _language is None:
        _language = resolve(os.environ.get(ENVIRONMENT_KEY))
    return _language


def tr(text: str, **values) -> str:
    translated = _catalog(language()).get(text, text)
    return translated.format(**values) if values else translated


def install_qt_translations(app) -> None:
    """Translate Qt's own dialogs and buttons (file dialogs, Yes/No, Cancel)."""
    if language() == "en":
        return
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

    translator = QTranslator(app)
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load(QLocale(language()), "qtbase", "_", path):
        app.installTranslator(translator)
        app._portrait_translator = translator  # keep it alive with the application
