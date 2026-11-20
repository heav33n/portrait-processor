import threading

import pytest
from portrait_processor.ai import Cancelled
from portrait_processor.calibration import Measurement, choose, load, machine_key, measure, save
from portrait_processor.levels import level
from PySide6.QtCore import QSettings


def costs(level1):
    return {n: level1 * level(n).cost for n in range(1, 5)}


def test_choice_is_the_most_detailed_bundled_level_within_budget():
    expected = max(n for n, s in costs(4.0).items() if s <= 20)
    chosen, seconds, fits = choose(4.0, 20)
    assert (chosen.number, fits) == (expected, True)
    assert seconds == pytest.approx(costs(4.0)[expected])


def test_auto_never_picks_the_download_only_level():
    assert choose(0.01, 10_000)[0].number == 4


def test_budget_below_level_one_still_answers_with_level_one():
    chosen, seconds, fits = choose(30.0, 5)
    assert (chosen.number, seconds, fits) == (1, 30.0, False)


def test_measure_confirms_and_steps_down_when_slower_than_predicted():
    calls = []
    actual = {1: 2.0, 2: 2.0 * level(2).cost, 3: 100.0, 4: 200.0}

    def run(number):
        calls.append(number)
        return actual[number]

    budget = 2.0 * level(4).cost + 1
    result = measure(run, budget, lambda _: None, threading.Event())
    assert calls[:2] == [1, 1]
    assert 4 in calls and 3 in calls
    chosen, _seconds, fits = result.choice(budget)
    assert chosen.number == 2 and fits
    assert result.confirmed[4] == 200.0


def test_cancel_raises_and_leaves_the_saved_measurement(tmp_path):
    preferences = QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat)
    old = Measurement(machine_key(), 3.0, {}, "2026-11-01")
    save(preferences, old)
    cancel = threading.Event()

    def run(number):
        cancel.set()
        return 1.0

    with pytest.raises(Cancelled):
        measure(run, 20, lambda _: None, cancel)
    assert load(preferences) == old


def test_saved_measurement_belongs_to_this_machine_only(tmp_path):
    preferences = QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat)
    save(preferences, Measurement("other cpu", 3.0, {3: 9.0}, "2026-11-20"))
    assert load(preferences) is None
    mine = Measurement(machine_key(), 3.0, {3: 9.0}, "2026-11-20")
    save(preferences, mine)
    assert load(preferences) == mine


def test_prediction_scales_only_the_mask_step_not_the_fixed_overhead():
    # Review finding #2: 1.07 s total of which 0.84 s is the mask; level 4's mask is 20x.
    m = Measurement(machine_key(), 1.07, {}, "2026-11-20", level1_mask=0.84)
    assert m.estimate(4) == pytest.approx(0.23 + 0.84 * level(4).cost)
    chosen, _seconds, fits = m.choice(0.23 + 0.84 * level(4).cost + 0.1)
    assert chosen.number == 4 and fits


def test_measure_records_the_mask_share_reported_by_the_runner():
    result = measure(lambda n: (2.0, 1.5), 3, lambda _: None, threading.Event())
    assert result.level1 == 2.0 and result.level1_mask == 1.5


@pytest.mark.parametrize(
    "stored",
    [
        '{"key": "%s", "level1": 2, "confirmed": [1, 2], "date": "x"}',
        '{"key": "%s", "level1": 2, "confirmed": {"9": 3.0}, "date": "x"}',
        "[]",
    ],
)
def test_damaged_or_outdated_measurement_never_breaks_startup(tmp_path, stored):
    preferences = QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat)
    preferences.setValue(
        "ai/measurement", stored.replace("%s", machine_key()) if "%s" in stored else stored
    )
    result = load(preferences)
    if result is not None:
        assert all(
            n in {lv.number for lv in __import__("portrait_processor.levels").levels.LEVELS}
            for n in result.confirmed
        )
        result.choice(20)
