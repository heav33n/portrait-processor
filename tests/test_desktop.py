import shutil
import threading

import numpy as np
import pytest
from PIL import Image
from portrait_processor.app import STYLE, Window
from portrait_processor.document import Face
from portrait_processor.document import render_document as real_render
from PySide6.QtCore import QSettings, Qt


class FakeEngine:
    def __init__(self, faces=True, fail=False):
        self.faces = faces
        self.fail = fail
        self.calls = 0

    def analyze(self, rgb, original_alpha, progress, cancel):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Model download unavailable")
        alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
        alpha[40:560, 160:640] = 255
        faces = [Face((399.5, 220), (330, 160), (470, 160))] if self.faces else []
        return alpha, faces, []

    def close(self):
        pass


@pytest.fixture
def window(qtbot, qapp, tmp_path):
    qapp.setStyle("Fusion")
    qapp.setStyleSheet(STYLE)
    widget = Window(QSettings(str(tmp_path / "test-settings.ini"), QSettings.Format.IniFormat))
    widget.engine = FakeEngine()

    def cleanup(widget):
        widget.timer.stop()
        qtbot.waitUntil(lambda: not widget.tasks, timeout=15000)
        for doc in widget.documents.values():
            doc.dirty = False
        widget.draft_dirty = {path: False for path in widget.draft_dirty}

    qtbot.addWidget(widget, before_close_func=cleanup)
    widget.show()
    yield widget


def wait_for_preview(qtbot, window):
    qtbot.waitUntil(
        lambda: (
            window.doc is not None
            and window.doc.result is not None
            and window.doc.result_revision == window.doc.revision
        ),
        timeout=15000,
    )


def test_process_all_prepares_queue_without_repeating_ai(qtbot, window, portrait):
    paths = [portrait.path]
    for i in range(4):
        path = portrait.path.with_name(f"Batch_{i}.png")
        shutil.copyfile(portrait.path, path)
        paths.append(path)
    window.open_paths(paths, prepare_all=True)
    wait_for_preview(qtbot, window)
    assert window.engine.calls == 5
    assert len(window.documents) == 1
    assert len(window.drafts) == 4
    assert "5/5 processed" in window.queue_hint.text()
    window.queue.setCurrentRow(4)
    wait_for_preview(qtbot, window)
    assert window.doc.path == paths[4]
    assert window.engine.calls == 5
    window._process_all()
    qtbot.waitUntil(lambda: not window.analysis_busy, timeout=15000)
    wait_for_preview(qtbot, window)
    assert window.engine.calls == 5


def test_cancel_batch_keeps_completed_work_and_can_resume(qtbot, window, portrait):
    from portrait_processor.ai import Cancelled

    class CancelSecond(FakeEngine):
        def analyze(self, rgb, original_alpha, progress, cancel):
            if self.calls == 1:
                self.calls += 1
                raise Cancelled("Batch cancelled")
            return super().analyze(rgb, original_alpha, progress, cancel)

    paths = [portrait.path, portrait.path.with_name("Second.png")]
    shutil.copyfile(paths[0], paths[1])
    window.engine = CancelSecond()
    window.open_paths(paths, prepare_all=True)
    wait_for_preview(qtbot, window)
    assert window.engine.calls == 2
    assert window.doc.path == paths[0]
    assert "1/2 processed" in window.queue_hint.text()
    window.engine = FakeEngine()
    window._process_all()
    qtbot.waitUntil(lambda: len(window.prepared) == 2 and window.batch_job is None, timeout=15000)
    wait_for_preview(qtbot, window)
    assert window.engine.calls == 1, "The completed mask must be reused"
    assert "2/2 processed" in window.queue_hint.text()


def test_missing_batch_face_does_not_stop_other_portraits(qtbot, window, portrait):
    from portrait_processor.document import Face

    class MissingFirst(FakeEngine):
        def analyze(self, *args):
            alpha, faces, warnings = super().analyze(*args)
            return alpha, [] if self.calls == 1 else faces, warnings

    second = portrait.path.with_name("Second.png")
    shutil.copyfile(portrait.path, second)
    window.engine = MissingFirst()
    window.open_paths([portrait.path, second], prepare_all=True)
    qtbot.waitUntil(lambda: window.doc is not None and not window.analysis_busy, timeout=15000)
    assert window.engine.calls == 2
    assert not window.doc.ready
    assert "1 need manual markers" in window.queue_hint.text()
    assert window.queue.item(0).text().startswith("! ")
    assert window.manual_button.isEnabled()
    window.doc.faces = [Face((399.5, 220), (330, 160), (470, 160))]
    window.doc.choose_face(0)
    window._activate(window.doc)
    wait_for_preview(qtbot, window)


def test_scan_import_queues_crops_and_keeps_output_outside_temp_folder(qtbot, window, portrait):
    from pathlib import Path

    from portrait_processor.scans import Scan

    rgb = np.tile(portrait.rgb, (1, 2, 1))
    h, w = portrait.rgb.shape[:2]
    crops = [np.float32([[x, 0], [x + w - 1, 0], [x + w - 1, h - 1], [x, h - 1]]) for x in (0, w)]
    scan = Scan(portrait.path, rgb, crops)
    window._import_scan(scan, crops)
    wait_for_preview(qtbot, window)
    assert len(window.paths) == 2
    assert window.engine.calls == 2
    assert window.doc.analyzed
    assert window.ai_button.text() == "Retry AI"
    assert window.output == portrait.path.parent / "processed_photos"
    assert window.paths[0].name == "portrait_01.png"
    directory = Path(window.scan_directory.name)
    assert directory.is_dir()
    window.export_name.setText("First scan portrait.jpg")
    window._save(True)
    qtbot.waitUntil(lambda: window.doc is not None and window.doc.path == window.paths[1])
    wait_for_preview(qtbot, window)
    assert (window.output / "First scan portrait.jpg").is_file()
    assert window.engine.calls == 2
    for doc in window.documents.values():
        doc.dirty = False
    window.draft_dirty = {p: False for p in window.draft_dirty}
    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    assert not directory.exists()


class GatedEngine(FakeEngine):
    def __init__(self, *blocked_calls):
        super().__init__()
        self.gates = {n: threading.Event() for n in blocked_calls}
        self.started = {n: threading.Event() for n in blocked_calls}
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def analyze(self, rgb, original_alpha, progress, cancel):
        from portrait_processor.ai import check_cancel

        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            number = self.calls + 1
            if number in self.gates:
                self.started[number].set()
                while not self.gates[number].wait(0.01):
                    check_cancel(cancel)
            return super().analyze(rgb, original_alpha, progress, cancel)
        finally:
            with self.lock:
                self.active -= 1

    def release(self):
        for event in self.gates.values():
            event.set()


def batch_paths(portrait, count=11):
    paths = []
    for i in range(count):
        path = portrait.path.with_name(f"Portrait_{i:02}.png")
        shutil.copyfile(portrait.path, path)
        paths.append(path)
    return paths


def test_background_preparation_keeps_render_save_and_cached_edits_responsive(
    qtbot, window, portrait
):
    paths = batch_paths(portrait)
    engine = GatedEngine(6)
    window.engine = engine
    window.open_paths(paths, prepare_all=True, auto_analyze=False, batch_mode=True)
    try:
        wait_for_preview(qtbot, window)
        qtbot.waitUntil(engine.started[6].is_set)
        assert len(window.prepared) == 5
        assert not window.analysis_busy and window.batch_job is not None
        assert not (portrait.path.parent / "processed_photos").exists()
        window.export_name.setText("My first five")
        window.export_format.setCurrentIndex(window.export_format.findData("JPEG"))
        window.horizontal_position.setValue(28)
        window.vertical_position.setValue(14)
        wait_for_preview(qtbot, window)
        assert window.save_button.isEnabled()
        window._save()
        assert (window.output / "My first five.jpg").is_file()
        for index in (1, 2, 3, 4, 0):
            window.queue.setCurrentRow(index)
            qtbot.waitUntil(
                lambda index=index: window.doc is not None and window.doc.path == paths[index]
            )
            wait_for_preview(qtbot, window)
        assert window.export_name.text() == "My first five"
        assert window.export_format.currentData() == "JPEG"
        assert window.doc.settings.position_x == 28 and window.doc.settings.position_y == 14
        assert not window.doc.dirty
        assert len(window.documents) <= 3
        engine.release()
        qtbot.waitUntil(
            lambda: len(window.prepared) == 11 and window.batch_job is None, timeout=15000
        )
        assert window.doc.path == paths[0], (
            "Background completion must not change the review selection"
        )
        assert window.doc.output_name == "My first five"
        assert window.doc.settings.position_x == 28
        assert not window.doc.dirty
        assert len(list(window.output.iterdir())) == 1, "Background work must not save photos"
        assert engine.calls == 11 and engine.max_active == 1
    finally:
        engine.release()


def test_catching_up_waits_only_until_the_requested_portrait_is_ready(qtbot, window, portrait):
    paths = batch_paths(portrait, 10)
    engine = GatedEngine(6, 7)
    window.engine = engine
    window.open_paths(paths, prepare_all=True, auto_analyze=False, batch_mode=True)
    try:
        wait_for_preview(qtbot, window)
        qtbot.waitUntil(engine.started[6].is_set)
        window.queue.setCurrentRow(4)
        qtbot.waitUntil(lambda: window.doc is not None and window.doc.path == paths[4])
        wait_for_preview(qtbot, window)
        window._save(True)
        assert window.waiting_path == paths[5] and window.analysis_busy
        assert not window.save_button.isEnabled()
        assert window.queue.isEnabled(), (
            "Already-prepared photos must remain reachable while waiting"
        )
        engine.gates[6].set()
        qtbot.waitUntil(engine.started[7].is_set)
        qtbot.waitUntil(lambda: window.doc is not None and window.doc.path == paths[5])
        wait_for_preview(qtbot, window)
        assert window.save_button.isEnabled() and not window.analysis_busy
        assert window.batch_job is not None and len(window.prepared) == 6
        window.export_name.setText("Sixth portrait")
        window.horizontal_position.setValue(12)
        wait_for_preview(qtbot, window)
        engine.release()
        qtbot.waitUntil(lambda: window.batch_job is None, timeout=15000)
        assert window.doc.path == paths[5]
        assert window.doc.output_name == "Sixth portrait"
        assert window.doc.settings.position_x == 12
    finally:
        engine.release()


def test_prepared_photos_remain_available_when_background_preparation_is_paused(
    qtbot, window, portrait
):
    paths = batch_paths(portrait)
    engine = GatedEngine(6)
    window.engine = engine
    window.open_paths(paths, prepare_all=True, auto_analyze=False, batch_mode=True)
    try:
        wait_for_preview(qtbot, window)
        qtbot.waitUntil(engine.started[6].is_set)
        window.export_name.setText("Keep my edits")
        window._cancel()
        qtbot.waitUntil(lambda: window.batch_job is None, timeout=5000)
        assert window.prefetch_paused and len(window.prepared) == 5
        assert window.doc.path == paths[0]
        assert window.export_name.text() == "Keep my edits"
        assert window.save_button.isEnabled()
        assert window.ai_mode.isEnabled()
        assert window.batch_button.text() == "Resume preparation"
        engine.release()
        window.batch_button.click()
        qtbot.waitUntil(
            lambda: len(window.prepared) == 11 and window.batch_job is None, timeout=15000
        )
        assert window.doc.path == paths[0]
        assert window.export_name.text() == "Keep my edits"
    finally:
        engine.release()


def test_drag_moves_portrait_without_ai_or_scaling(qtbot, window, portrait):
    from PySide6.QtCore import QPointF

    window.open_paths([portrait.path], auto_analyze=False)
    wait_for_preview(qtbot, window)
    initial = window.doc.result
    window.snap.setChecked(False)
    assert window.canvas.tool == "portrait"
    start = window.canvas.mapFromScene(QPointF(473, 700))
    finish = window.canvas.mapFromScene(QPointF(525, 735))
    qtbot.mousePress(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    assert not window.save_button.isEnabled()
    qtbot.mouseMove(window.canvas.viewport(), finish)
    qtbot.mouseRelease(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=finish)
    wait_for_preview(qtbot, window)
    assert window.engine.calls == 0
    dx, dy = window.doc.settings.position_x, window.doc.settings.position_y
    assert abs(dx - 52) < 5 and abs(dy - 35) < 5
    assert window.doc.result["scale"] == initial["scale"]
    np.testing.assert_allclose(window.doc.result["transform"][2], initial["transform"][2] + dx)
    np.testing.assert_allclose(window.doc.result["transform"][5], initial["transform"][5] + dy)
    window._reset_position()
    wait_for_preview(qtbot, window)
    np.testing.assert_array_equal(window.doc.result["rgb"], initial["rgb"])


def test_snap_can_be_bypassed_during_portrait_drag(qtbot, window, portrait):
    window.open_paths([portrait.path], auto_analyze=False)
    wait_for_preview(qtbot, window)
    window.snap.setChecked(True)
    window._begin_position()
    window._move_position(3, 3)
    assert window.doc.settings.position_x == 0
    assert window.doc.settings.position_y == 0
    window._move_position(3, 3, bypass=True)
    assert window.doc.settings.position_x == 3
    assert window.doc.settings.position_y == 3
    window._end_position()
    wait_for_preview(qtbot, window)


def test_open_review_change_spacing_and_save(qtbot, window, portrait):
    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    assert window.save_button.isEnabled()
    assert window.engine.calls == 1
    window.headroom.setValue(100)
    assert not window.save_button.isEnabled()
    wait_for_preview(qtbot, window)
    assert window.engine.calls == 1, "Spacing must reuse the AI mask"
    qtbot.mouseClick(window.save_button, Qt.MouseButton.LeftButton)
    destination = portrait.path.parent / "processed_photos" / "portrait.png"
    assert destination.is_file()
    with Image.open(destination) as image:
        assert image.size == (948, 1048)
        np.testing.assert_array_equal(np.asarray(image), window.doc.result["rgb"])
    assert not window.doc.dirty


def test_missing_face_requires_manual_markers(qtbot, window, portrait):
    window.engine.faces = False
    window.open_paths([portrait.path])
    qtbot.waitUntil(lambda: window.doc is not None and window.doc.analyzed, timeout=5000)
    assert not window.save_button.isEnabled()
    for name, point in [("nose", (399.5, 220)), ("eye_a", (330, 160)), ("eye_b", (470, 160))]:
        window._tool_changed(name)
        window._set_point(*point)
    wait_for_preview(qtbot, window)
    assert window.save_button.isEnabled()


def test_failed_ai_allows_manual_editing(qtbot, window, portrait):
    window.engine.fail = True
    window.open_paths([portrait.path])
    qtbot.waitUntil(lambda: window.doc is not None and not window.analysis_busy, timeout=5000)
    assert "unavailable" in window.status.text()
    assert not window.save_button.isEnabled()
    qtbot.mouseClick(window.manual_button, Qt.MouseButton.LeftButton)
    assert window.doc.analyzed
    assert window.canvas.tool == "nose"


def test_preview_controls_do_not_change_exports(qtbot, window, portrait):
    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    expected = window.doc.result["rgb"].copy()
    revision = window.doc.revision
    window.view_mode.setCurrentIndex(1)
    window.guide.setChecked(False)
    window.show_mask.setChecked(True)
    assert window.doc.revision == revision
    np.testing.assert_array_equal(window.doc.result["rgb"], expected)


def test_brush_stroke_prevents_stale_save_and_can_be_undone(qtbot, window, portrait):
    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    window._tool_changed("erase")
    window._begin_stroke()
    window._paint(400, 300)
    assert not window.save_button.isEnabled()
    window._end_stroke()
    assert window.doc.alpha[300, 400] == 0
    window._undo()
    assert window.doc.alpha[300, 400] == 255
    wait_for_preview(qtbot, window)


def test_save_next_keeps_previous_edits_without_repeating_ai(qtbot, window, portrait):
    second = portrait.path.with_name("second.png")
    shutil.copyfile(portrait.path, second)
    window.open_paths([portrait.path, second])
    wait_for_preview(qtbot, window)
    first_doc = window.doc
    window._save(True)
    qtbot.waitUntil(lambda: window.doc is not None and window.doc.path == second, timeout=5000)
    wait_for_preview(qtbot, window)
    assert not first_doc.dirty
    assert window.engine.calls == 2
    window.queue.setCurrentRow(0)
    assert window.doc is first_doc
    assert window.save_button.isEnabled()
    assert window.engine.calls == 2


def test_old_worker_result_cannot_replace_newer_settings(qtbot, window, portrait, monkeypatch):
    from portrait_processor import app

    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    released = threading.Event()
    started = threading.Event()

    def slow_render(doc):
        started.set()
        assert released.wait(5)
        return real_render(doc)

    monkeypatch.setattr(app, "render_document", slow_render)
    window.headroom.setValue(90)
    qtbot.waitUntil(started.is_set, timeout=3000)
    window.headroom.setValue(130)
    assert not window.save_button.isEnabled()
    released.set()
    wait_for_preview(qtbot, window)
    ys, _ = np.nonzero(window.doc.result["alpha"])
    assert ys.min() >= 130


def test_large_queue_reloads_drafts_without_repeating_ai(qtbot, window, portrait):
    paths = [portrait.path]
    for i in range(4):
        path = portrait.path.with_name(f"copy_{i}.png")
        shutil.copyfile(portrait.path, path)
        paths.append(path)
    window.open_paths(paths)
    wait_for_preview(qtbot, window)
    window.export_name.setText("My first portrait")
    for i in range(1, len(paths)):
        window.queue.setCurrentRow(i)
        qtbot.waitUntil(lambda: window.doc is not None and not window.analysis_busy, timeout=5000)
        wait_for_preview(qtbot, window)
    assert len(window.documents) <= 3
    assert paths[0] in window.drafts
    assert window.engine.calls == len(paths)
    window.queue.setCurrentRow(0)
    qtbot.waitUntil(lambda: window.doc is not None and not window.analysis_busy, timeout=5000)
    wait_for_preview(qtbot, window)
    assert window.doc.path == paths[0]
    assert window.export_name.text() == "My first portrait"
    assert window.engine.calls == len(paths)


def test_save_as_input_names_the_export_without_repeating_processing(qtbot, window, portrait):
    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    revision = window.doc.revision
    window.export_name.setText("Alice.png")
    assert window.doc.revision == revision
    assert window.save_button.isEnabled()
    qtbot.mouseClick(window.save_button, Qt.MouseButton.LeftButton)
    assert window.doc.saved_path.name == "Alice.png"
    assert window.engine.calls == 1
    window.export_name.setText("../elsewhere")
    assert not window.save_button.isEnabled()
    window.export_name.setText("")
    assert not window.save_button.isEnabled()


def test_frame_limits_update_preview_without_repeating_ai(qtbot, window, portrait):
    window.preferences.setValue("headroom", 60)  # The old top-anchored preference is obsolete.
    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    assert window.headroom.value() == 180
    assert window.side_padding.value() == 64
    assert window.bottom_gap.value() == 0
    window.side_padding.setValue(160)
    window.bottom_gap.setValue(40)
    window.headroom.setValue(340)
    wait_for_preview(qtbot, window)
    ys, xs = np.nonzero(window.doc.result["alpha"])
    assert ys.min() >= 340 and ys.max() == 1007
    assert xs.min() >= 160 and xs.max() < 788
    assert len(window.canvas.frame_guides) == 4
    assert window.canvas.frame_guides[0].line().y1() == 339.5
    assert window.engine.calls == 1
    window.guide.setChecked(False)
    assert not window.canvas.frame_guides


def test_hair_preservation_is_default_and_strict_trim_is_optional(qtbot, window, portrait):
    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    assert not window.strict_symmetry.isChecked()
    window.doc.alpha[100:200, 90:160] = 255
    window.doc.changed()
    window._schedule_render()
    wait_for_preview(qtbot, window)
    np.testing.assert_array_equal(window.doc.result["alpha"], window.doc.result["before_alpha"])
    window.strict_symmetry.setChecked(True)
    wait_for_preview(qtbot, window)
    assert window.doc.settings.strict_symmetry
    assert window.doc.result["removed_fraction"] > 0
    assert np.array_equal(window.doc.result["alpha"], window.doc.result["alpha"][:, ::-1])
    window.strict_symmetry.setChecked(False)
    wait_for_preview(qtbot, window)
    assert not window.doc.settings.strict_symmetry
    np.testing.assert_array_equal(window.doc.result["alpha"], window.doc.result["before_alpha"])
    assert window.engine.calls == 1


def test_jpeg_selection_and_quality_do_not_repeat_processing(qtbot, window, portrait):
    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    revision = window.doc.revision
    window.export_name.setText("My portrait.jpg")
    assert window.export_format.currentData() == "JPEG"
    assert window.export_quality.isVisible()
    window.export_quality.setValue(86)
    assert window.doc.revision == revision
    window._save()
    with Image.open(window.doc.saved_path) as image:
        assert image.format == "JPEG" and image.size == (948, 1048)
    window.export_format.setCurrentIndex(window.export_format.findData("TIFF"))
    assert not window.export_quality.isVisible()
    window._save()
    assert window.doc.saved_path.suffix == ".tiff"
    assert window.engine.calls == 1


def test_processed_photo_can_be_repositioned_and_manually_edited(qtbot, window, portrait):
    window.open_paths([portrait.path])
    wait_for_preview(qtbot, window)
    window.horizontal_position.setValue(75)
    wait_for_preview(qtbot, window)
    t = window.doc.result["transform"]
    x, y = window.doc.nose
    assert abs(t[0] * x + t[1] * y + t[2] - 548.5) < 1e-6
    before = window.doc.alpha.copy()
    assert window.manual_button.isEnabled()
    window.manual_button.click()
    assert window.canvas.tool == "nose"
    np.testing.assert_array_equal(window.doc.alpha, before)
    wait_for_preview(qtbot, window)
    assert window.engine.calls == 1


@pytest.mark.parametrize("focus", ["export_name", "headroom"])
def test_enter_saves_and_moves_to_the_next_photo(qtbot, window, portrait, focus):
    second = portrait.path.with_name("second.png")
    shutil.copyfile(portrait.path, second)
    window.open_paths([portrait.path, second])
    wait_for_preview(qtbot, window)
    first_doc = window.doc
    field = getattr(window, focus)
    window.activateWindow()
    field.setFocus()
    qtbot.waitUntil(field.hasFocus, timeout=2000)
    qtbot.keyClick(field, Qt.Key.Key_Return)
    qtbot.waitUntil(lambda: window.doc is not None and window.doc.path == second, timeout=5000)
    assert not first_doc.dirty
    assert "Enter" in window.next_button.toolTip()
