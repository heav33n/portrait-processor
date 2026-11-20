# ruff: noqa: F811 - tests reuse the shared "window" fixture name
import numpy as np
import pytest
from portrait_processor.app import Window
from portrait_processor.calibration import Measurement, load, machine_key, save
from PySide6.QtCore import QSettings
from test_desktop import (  # pyright: ignore[reportImplicitRelativeImport]
    window,  # noqa: F401 - shared fixture
)


class MeasuringEngine:
    def __init__(self, seconds):
        self.seconds = seconds
        self.level = 1
        self.measured = []

    def measure_level(self, level, rgb, alpha, progress, cancel):
        self.measured.append(level)
        assert rgb.ndim == 3 and alpha.shape == rgb.shape[:2]
        return self.seconds[level]

    def analyze(self, rgb, alpha, progress, cancel):
        return alpha, [], []

    def close(self):
        pass


def make(qtbot, tmp_path, **values):
    preferences = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    for key, value in values.items():
        preferences.setValue(key.replace("__", "/"), value)
    window = Window(preferences)
    qtbot.addWidget(window, before_close_func=lambda w: qtbot.waitUntil(lambda: not w.tasks))
    return window, preferences


def test_fresh_install_starts_in_auto_with_a_20_second_budget(qtbot, tmp_path):
    window, _ = make(qtbot, tmp_path)
    assert window.ai_mode.currentData() == "auto"
    assert window.ai_budget.value() == 20
    assert window.engine.level == 1
    assert "Measure this PC" in window.ai_estimate.text()


@pytest.mark.parametrize("model,level", [("birefnet-portrait", 5), ("u2net_human_seg", 1)])
def test_old_model_preference_migrates_to_a_fixed_level(qtbot, tmp_path, model, level):
    window, _ = make(qtbot, tmp_path, ai__model=model)
    assert window.ai_mode.currentData() == level
    assert window.engine.level == level
    assert not window.ai_budget.isVisible()


def test_budget_change_repicks_from_the_stored_measurement_without_running(qtbot, tmp_path):
    preferences = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    save(preferences, Measurement(machine_key(), 2.0, {}, "2026-11-20"))
    window, _ = make(qtbot, tmp_path)
    engine = MeasuringEngine({})
    window.engine = engine  # pyright: ignore[reportAttributeAccessIssue]
    window.ai_budget.setValue(5)
    assert engine.level == 2 and engine.measured == []
    window.ai_budget.setValue(600)
    assert engine.level == 4
    assert "Detailed" in window.ai_estimate.text()
    window.ai_budget.setValue(1)
    assert window.engine.level == 1
    assert "over" in window.ai_estimate.text()


def test_measure_this_pc_runs_in_the_background_and_saves(qtbot, tmp_path):
    window, preferences = make(qtbot, tmp_path)
    engine = MeasuringEngine({1: 1.0, 2: 1.3, 3: 4.0, 4: 30.0})
    window.engine = engine  # pyright: ignore[reportAttributeAccessIssue]
    window.ai_budget.setValue(25)
    window.measure_button.click()
    qtbot.waitUntil(lambda: load(preferences) is not None and not window.tasks, timeout=5000)
    assert engine.measured[:2] == [1, 1] and 4 in engine.measured
    assert window.engine.level == 3
    assert "Balanced" in window.ai_estimate.text()


def test_fixed_level_hides_auto_controls(qtbot, tmp_path):
    window, preferences = make(qtbot, tmp_path)
    window.show()
    assert window.ai_budget.isVisible()
    window.ai_mode.setCurrentIndex(window.ai_mode.findData(3))
    assert not window.ai_budget.isVisible() and window.engine.level == 3
    assert preferences.value("ai/level") == "3"
    assert np  # keep numpy imported for engine arrays


class SlowMeasure(MeasuringEngine):
    def __init__(self):
        import threading

        super().__init__({})
        self.gate = threading.Event()

    def measure_level(self, level, rgb, alpha, progress, cancel):
        self.measured.append(level)
        while not self.gate.wait(0.05):
            if cancel.is_set():
                break
        return 1.0


def test_review_1_controls_stay_locked_while_measuring_even_when_picking_photos(
    qtbot, window, portrait
):
    from test_desktop import (  # pyright: ignore[reportImplicitRelativeImport]
        GatedEngine,
        batch_paths,
        wait_for_preview,
    )

    paths = batch_paths(portrait)
    engine = GatedEngine(6)
    window.engine = engine
    window.open_paths(paths, prepare_all=True, auto_analyze=False, batch_mode=True)
    wait_for_preview(qtbot, window)
    qtbot.waitUntil(engine.started[6].is_set)
    window._cancel()
    engine.release()
    qtbot.waitUntil(lambda: window.batch_job is None, timeout=5000)
    measuring = SlowMeasure()
    window.engine = measuring
    window._select_photo(7)
    if window.waiting_path is not None:
        window.measure_pc()  # refused while a photo waits for its AI result
        assert window.analysis_job is None and not measuring.measured
        window._select_photo(0)
    window.measure_pc()
    first = window.analysis_job
    assert first is not None
    window._select_photo(7)
    window._select_photo(8)
    assert window.waiting_path is None
    assert window.analysis_busy
    assert not window.ai_mode.isEnabled() and not window.measure_button.isEnabled()
    assert window.cancel_button.isVisible()
    window.measure_pc()
    assert window.analysis_job == first
    measuring.gate.set()
    qtbot.waitUntil(lambda: not window.tasks, timeout=10000)
    assert measuring.measured.count(1) == 2


def test_review_5_cancelling_a_measurement_keeps_preparation_running(qtbot, window, portrait):
    started = []
    window.open_paths([portrait.path], prepare_all=False)
    qtbot.waitUntil(lambda: not window.tasks, timeout=15000)
    assert not window.prefetch_paused
    measuring = SlowMeasure()
    window.engine = measuring
    window._start_prefetch = lambda: started.append(True)
    window.measure_pc()
    window._cancel()
    qtbot.waitUntil(lambda: not window.tasks, timeout=10000)
    assert not window.prefetch_paused
    assert "Measurement cancelled" in window.status.text()
    assert started


def test_review_7_untested_levels_are_labelled_as_predicted(qtbot, tmp_path):
    preferences = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    save(preferences, Measurement(machine_key(), 2.0, {3: 6.0}, "2026-11-20", 1.5))
    window, _ = make(qtbot, tmp_path)
    window.ai_budget.setValue(6)
    assert window.engine.level == 3 and "predicted" not in window.ai_estimate.text()
    window.ai_budget.setValue(600)
    assert window.engine.level == 4 and "predicted" in window.ai_estimate.text()
