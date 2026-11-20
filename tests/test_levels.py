import pytest
from portrait_processor.levels import LEVELS, level, level_from_preference


def test_catalog_order_and_bundling():
    assert [lv.number for lv in LEVELS] == [1, 2, 3, 4, 5]
    assert level(1).model == "u2net_human_seg" and not level(1).refine
    assert level(2).model == "u2net_human_seg" and level(2).refine
    assert level(5).model == "birefnet-portrait"
    assert {lv.number for lv in LEVELS if lv.bundled} == {1, 2, 3, 4}
    costs = [lv.cost for lv in LEVELS]
    assert costs[0] == 1.0 and costs == sorted(set(costs))


@pytest.mark.parametrize(
    "value,expected",
    [
        ("u2net_human_seg", 1),
        ("birefnet-portrait", 5),
        ("3", 3),
        (3, 3),
        ("auto", None),
        ("9", None),
        ("", None),
    ],
)
def test_preferences_map_to_levels(value, expected):
    assert level_from_preference(value) == expected


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError):
        level(0)
