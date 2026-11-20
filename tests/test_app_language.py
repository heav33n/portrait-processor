import pytest
from portrait_processor import i18n
from portrait_processor.app import Window
from PySide6.QtCore import QSettings


@pytest.fixture
def restore_language():
    previous = i18n._language
    yield
    i18n.set_language(previous or "en")


def make(qtbot, tmp_path):
    preferences = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    window = Window(preferences)
    qtbot.addWidget(window, before_close_func=lambda w: qtbot.waitUntil(lambda: not w.tasks))
    return window, preferences


def test_language_choice_is_saved_and_applies_on_next_start(qtbot, tmp_path, restore_language):
    i18n.set_language("en")
    window, preferences = make(qtbot, tmp_path)
    assert window.language.currentData() == "auto"
    window.language.setCurrentIndex(window.language.findData("es"))
    assert preferences.value("ui/language") == "es"
    assert "next time you open" in window.status.text()
    # The running window keeps its language until restart.
    assert window.ai_button.text() == "Retry AI"
    assert make(qtbot, tmp_path)[0].language.currentData() == "es"


def test_window_opens_in_spanish(qtbot, tmp_path, restore_language):
    i18n.set_language("es")
    window, _ = make(qtbot, tmp_path)
    assert window.ai_button.text() == "Reintentar IA"
    assert window.measure_button.text() == "Medir esta PC"
    assert window.ai_mode.itemText(1) == "1 · Más rápido"
