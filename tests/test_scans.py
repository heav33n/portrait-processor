import threading

import numpy as np
import pytest
from PIL import Image, ImageDraw
from portrait_processor.ai import Cancelled
from portrait_processor.scans import (
    Scan,
    detect_crops,
    extract_crop,
    extract_scan,
    load_scan,
    ordered_corners,
    reading_order,
)


@pytest.fixture
def scanned_sheet(tmp_path):
    sheet = Image.new("RGB", (1040, 1460), (236, 237, 237))
    draw = ImageDraw.Draw(sheet)
    for row in range(5):
        for col in range(4):
            x, y = 25 + col * 255, 20 + row * 285
            draw.rectangle((x + 3, y + 3, x + 223, y + 263), fill=(200, 202, 201))
            draw.rectangle((x, y, x + 220, y + 260), fill=(245, 246, 245))
            draw.ellipse((x + 55, y + 30, x + 165, y + 185), fill=(50, 35, 25))
            draw.ellipse((x + 70, y + 60, x + 150, y + 180), fill=(180, 130, 110))
            draw.polygon(
                [(x + 85, y + 175), (x + 135, y + 175), (x + 220, y + 260), (x, y + 260)],
                fill=(40, 75 + row * 20, 50 + col * 30),
            )
    path = tmp_path / "Sheet.png"
    sheet.transpose(Image.Transpose.ROTATE_180).save(path)
    return path, np.asarray(sheet)


def test_rotates_whole_scan_then_finds_all_prints_in_reading_order(scanned_sheet):
    path, expected = scanned_sheet
    original = path.read_bytes()
    scan = load_scan(path)
    np.testing.assert_array_equal(scan.rgb, expected)
    assert path.read_bytes() == original
    assert len(scan.crops) == 20
    for i, crop in enumerate(scan.crops):
        np.testing.assert_allclose(
            crop.mean(axis=0), [135 + i % 4 * 255, 150 + i // 4 * 285], atol=6
        )


def test_extraction_keeps_complete_prints_and_uses_numbered_names(scanned_sheet, tmp_path):
    path, rgb = scanned_sheet
    crops = [
        np.float32([[25, 20], [245, 20], [245, 280], [25, 280]]),
        np.float32([[280, 20], [500, 20], [500, 280], [280, 280]]),
    ]
    output = tmp_path / "extracted"
    output.mkdir()
    paths = extract_scan(Scan(path, rgb, crops), list(reversed(crops)), output)
    assert [p.name for p in paths] == ["Sheet_01.png", "Sheet_02.png"]
    with Image.open(paths[0]) as image:
        assert image.width >= 219 and image.height >= 259
        a = np.asarray(image)
        assert a[8, 8].min() > 240  # Includes space above the head.
        assert a[-2, 110, 1] == 75  # Includes the bottom of the clothing.


def test_blank_sheet_does_not_invent_crops():
    assert detect_crops(np.full((600, 800, 3), 245, np.uint8)) == []


def test_cancel_does_not_extract_files(scanned_sheet, tmp_path):
    path, rgb = scanned_sheet
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        load_scan(path, cancel=cancel)
    with pytest.raises(Cancelled):
        extract_scan(Scan(path, rgb, []), [], tmp_path, cancel=cancel)


def test_degenerate_crop_is_rejected():
    with pytest.raises(ValueError):
        extract_crop(np.zeros((100, 100, 3), np.uint8), np.zeros((4, 2), np.float32))


def test_tilted_row_is_sorted_left_to_right():
    def box(x, y):
        return np.float32([[x, y], [x + 100, y], [x + 100, y + 130], [x, y + 130]])

    a, b, c = box(5, 15), box(140, 0), box(5, 200)
    actual = reading_order([c, b, a])
    for result, expected in zip(actual, [a, b, c], strict=True):
        np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(ordered_corners(a[[2, 3, 0, 1]]), a)


def test_scan_review_add_move_remove_and_rotate(qtbot, scanned_sheet):
    from portrait_processor.scan_dialog import ScanReview
    from PySide6.QtCore import QPointF, Qt

    path, rgb = scanned_sheet
    crop = np.float32([[25, 20], [245, 20], [245, 280], [25, 280]])
    dialog = ScanReview(Scan(path, rgb.copy(), [crop]))
    qtbot.addWidget(dialog)
    dialog.show()
    view = dialog.canvas
    qtbot.waitUntil(lambda: view.width() > 100)
    dialog.add_button.click()
    p1 = view.mapFromScene(QPointF(280, 20))
    p2 = view.mapFromScene(QPointF(500, 280))
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=p1)
    qtbot.mouseMove(view.viewport(), p2)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=p2)
    assert len(view.crops) == 2
    assert dialog.import_button.isEnabled()
    assert not dialog.add_button.isChecked()
    before = view.crops[1].copy()
    center = view.mapFromScene(QPointF(*before.mean(axis=0)))
    moved = view.mapFromScene(QPointF(*(before.mean(axis=0) + [30, 30])))
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=center)
    qtbot.mouseMove(view.viewport(), moved)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=moved)
    assert view.crops[1].mean(axis=0)[0] > before.mean(axis=0)[0] + 20
    dialog.remove_button.click()
    assert len(view.crops) == 1
    view.rotate()
    np.testing.assert_array_equal(view.scan.rgb, rgb[::-1, ::-1])
    view.rotate()
    np.testing.assert_array_equal(view.scan.rgb, rgb)
    np.testing.assert_allclose(view.crops[0], crop)
