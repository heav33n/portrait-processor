"""The review-first Linux desktop editor."""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSettings, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import _native, i18n
from .ai import Cancelled, check_cancel
from .calibration import load as load_measurement
from .calibration import measure as measure_levels
from .calibration import save as save_measurement
from .canvas import Canvas
from .document import (
    EXPORT_FORMATS,
    EXPORT_SUFFIXES,
    SUPPORTED,
    Document,
    export_stem,
    render_document,
    save_image,
)
from .i18n import tr
from .inference_process import InferenceProcess
from .levels import LEVELS, display_name, level_from_preference
from .preferences import bool_setting, int_setting
from .scan_dialog import ScanReview
from .scans import extract_scan, load_scan

LOG = logging.getLogger(__name__)
AUTO = "auto"  # Preference value for the automatic level and language.
PIXELS = " px"
IMAGE_PATTERNS = "*.jpg *.jpeg *.png *.webp *.tif *.tiff *.bmp"
LANGUAGE_CHOICES = ((AUTO, None), ("en", "English"), ("es", "Español"))


def format_badge(image_format):
    return f"948 × 1048  ·  {image_format}"


def ai_mode_tip():
    return tr(
        "Lower levels are faster with rougher edges; higher levels are slower with finer hair "
        "and outlines. Auto picks the best level that fits your time per photo."
    )


def calibration_photo():
    """A neutral portrait-sized image: every model resizes its input, so content barely matters."""
    h, w = 1800, 1500
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    head = ((x - w / 2) / (w * 0.28)) ** 2 + ((y - h * 0.42) / (h * 0.3)) ** 2 < 1
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[..., 0] = 120 + 80 * (x / w)
    rgb[..., 1] = 140 + 60 * (y / h)
    rgb[..., 2] = 170
    rgb[head] = (205, 160, 135)
    return rgb, np.full((h, w), 255, np.uint8)


STYLE = """
QWidget { color: #e4ece7; background: transparent; font-family: 'Segoe UI', 'Noto Sans', sans-serif; font-size: 13px; }
QMainWindow, QWidget#workspace, QDialog, QMessageBox, QFileDialog { background: #151c1b; }
QLabel#title { font-size: 25px; font-weight: 700; color: #e9f4ed; }
QLabel#subtitle, QLabel#muted { color: #9eafa5; font-size: 12px; }
QLabel#section { font-size: 11px; font-weight: 700; color: #93b4a0; }
QLabel#cardTitle { font-size: 14px; font-weight: 700; color: #d4e9dc; }
QLabel#badge { background: #283d31; color: #b1d5bd; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
QFrame#card { background: #1e2825; border: 1px solid #35483e; border-radius: 12px; }
QPushButton { background: #26342e; border: 1px solid #40554a; border-radius: 7px; padding: 8px 12px; min-height: 18px; font-weight: 500; }
QPushButton:hover { background: #30443a; border-color: #7b9f87; }
QPushButton:pressed { background: #385140; }
QPushButton:checked { background: #2f4839; color: #c2ebd2; border-color: #62866c; }
QPushButton:disabled { color: #718278; background: #202a25; border-color: #2b3730; }
QPushButton#primary { background: #9ed9b2; color: #172b1e; border-color: #9ed9b2; font-weight: 700; padding: 9px 18px; }
QPushButton#primary:hover { background: #b7e6c5; }
QPushButton#primary:pressed { background: #83bf97; }
QPushButton#primary:disabled { background: #2b4032; color: #73917e; border-color: #2b4032; }
QPushButton#disclosure { text-align: left; padding: 13px 16px; background: #1e2825; border-radius: 10px; font-weight: 600; }
QListWidget { background: transparent; border: none; outline: none; padding: 0; }
QListWidget::item { padding: 13px 10px; margin-bottom: 4px; border-radius: 7px; color: #b5c6bb; }
QListWidget::item:selected { background: #30493b; color: #cef1da; font-weight: 600; }
QListWidget::item:hover:!selected { background: #293a30; }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit { background: #18221e; border: 1px solid #40554a; border-radius: 7px; padding: 7px 10px; min-height: 20px; selection-background-color: #487b59; selection-color: white; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #8cc9a3; background: #1c2b23; }
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled { background: #1b2520; color: #718278; border-color: #2b3730; }
QSpinBox QLineEdit, QDoubleSpinBox QLineEdit { background: transparent; border: none; padding: 0; min-height: 0; border-radius: 0; }
QComboBox { padding-right: 24px; }
QComboBox QAbstractItemView { background: #1e2825; color: #e4ece7; selection-background-color: #30493b; selection-color: #cef1da; padding: 6px; border: 1px solid #40554a; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: #4b6556; border-radius: 2px; min-height: 35px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QProgressBar { border: none; background: #2b4032; border-radius: 3px; max-height: 4px; }
QProgressBar::chunk { background: #9ed9b2; }
QToolTip { background: #263c2e; color: #e4ece7; border: 1px solid #486d55; padding: 8px; }
QSplitter::handle { background: transparent; width: 16px; }

QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 15px; height: 15px; background: #18221e; border: 1px solid #789583; border-radius: 3px; }
QCheckBox::indicator:checked { background: #477b59; border-color: #77aa88; image: url("CHECK_ASSET"); }
QCheckBox::indicator:disabled { background: #1b2520; border-color: #35483e; }
""".replace("CHECK_ASSET", Path(__file__).with_name("check.svg").as_posix())


def dark_palette():
    """Cover native Qt controls and popups as well as the styled panels."""
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: "#151c1b",
        QPalette.ColorRole.WindowText: "#e4ece7",
        QPalette.ColorRole.Base: "#18221e",
        QPalette.ColorRole.AlternateBase: "#26342e",
        QPalette.ColorRole.Text: "#e4ece7",
        QPalette.ColorRole.PlaceholderText: "#86998e",
        QPalette.ColorRole.Button: "#26342e",
        QPalette.ColorRole.ButtonText: "#e4ece7",
        QPalette.ColorRole.Highlight: "#487b59",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.ToolTipBase: "#263c2e",
        QPalette.ColorRole.ToolTipText: "#e4ece7",
        QPalette.ColorRole.Link: "#9ed9b2",
    }.items():
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.PlaceholderText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#718278"))
    return palette


class TaskSignals(QObject):
    progress = Signal(object)
    complete = Signal(object)


class Task(QRunnable):
    def __init__(self, identifier, function):
        super().__init__()
        self.identifier = identifier
        self.function = function
        self.cancel = threading.Event()
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            result = self.function(
                lambda text: self.signals.progress.emit((self.identifier, text)), self.cancel
            )
            check_cancel(self.cancel)
            payload = (self.identifier, result, None)
        except Exception as error:
            if not isinstance(error, Cancelled):
                LOG.exception("Processing failed")
            payload = (self.identifier, None, error)
        self.signals.complete.emit(payload)


def button(text, callback, primary=False):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    if primary:
        widget.setObjectName("primary")
    return widget


def hint(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("muted")
    return label


def card(title):
    frame = QFrame()
    frame.setObjectName("card")
    body = QVBoxLayout(frame)
    body.setContentsMargins(16, 16, 16, 16)
    body.setSpacing(12)
    label = QLabel(title)
    label.setObjectName("cardTitle")
    body.addWidget(label)
    return frame, body


if TYPE_CHECKING:
    _SpinFieldBase = QAbstractSpinBox
else:
    _SpinFieldBase = object


class SpinFieldLayout(_SpinFieldBase):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the editor readable with Qt's styled, buttonless spin boxes.
        self.lineEdit().setGeometry(8, 1, max(1, self.width() - 16), self.height() - 2)


class IntegerField(SpinFieldLayout, QSpinBox):
    pass


class DecimalField(SpinFieldLayout, QDoubleSpinBox):
    pass


def stepper(spin):
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    row = QWidget()
    body = QHBoxLayout(row)
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(4)
    decrease = button("−", spin.stepDown)
    increase = button("+", spin.stepUp)
    for control, description in [
        (decrease, tr("Decrease value")),
        (increase, tr("Increase value")),
    ]:
        control.setFixedWidth(30)
        control.setToolTip(description)
        control.setAccessibleName(description)
    body.addWidget(decrease)
    body.addWidget(spin, 1)
    body.addWidget(increase)
    return row


class Window(QMainWindow):
    def __init__(self, preferences=None):
        super().__init__()
        self.setWindowTitle(tr("Portrait Processor"))
        self.resize(1380, 900)
        self.setPalette(dark_palette())
        self.setMinimumSize(1060, 720)
        self.preferences = (
            preferences
            if preferences is not None
            else QSettings("PortraitProcessor", "PortraitProcessor")
        )
        if self.preferences.value("ai/level") is None:
            # Migrate the former two-mode choice; new installs start in Auto.
            legacy = self.preferences.value("ai/model")
            number = level_from_preference(legacy) if legacy is not None else None
            self.preferences.setValue("ai/level", "auto" if number is None else str(number))
        self.engine = InferenceProcess(level=1)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.preparation_pool = QThreadPool(self)
        self.preparation_pool.setMaxThreadCount(1)
        self.tasks = {}
        self.serial = 0
        self.analysis_job = None
        self.measure_job = None
        self.analysis_busy = False
        self.batch_job = None
        self.batch_state: dict[str, Any] | None = None
        self.waiting_path = None
        self.last_review_path = None
        self.prefetch_enabled = False
        self.prefetch_paused = False
        self.render_running = False
        self.paths = []
        self.auto_analyze = True
        self.batch_mode = False
        self.prepared = set()
        self.manual_required = set()
        self.drag_state = None
        self.documents = {}
        self.session_directory = tempfile.TemporaryDirectory(prefix="portrait-session-")
        self.scan_directory = None
        self.drafts = {}
        self.draft_dirty = {}
        self.doc = None
        self.output = None
        self.stroke_before = None
        self.previous_brush = None
        self.pending_close = False
        self.syncing = False
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(160)
        self.timer.timeout.connect(self._render)
        self._shortcuts()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(1000)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.elapsed_text = None
        self._refresh_enabled()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        root.setObjectName("workspace")
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(16)
        self.setCentralWidget(root)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(4)
        title = QLabel(tr("Portrait Processor"))
        title.setObjectName("title")
        subtitle = QLabel(tr("A little more balance. A portrait that feels right."))
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        badge = QLabel(format_badge("PNG"))
        self.format_badge = badge
        badge.setObjectName("badge")
        header.addWidget(badge)
        header.addSpacing(12)
        self.open_button = button(tr("Add photos…"), self.open_photos)
        self.folder_button = button(tr("Open folder…"), self.open_folder)
        self.scan_button = button(tr("Import scan…"), self.open_scan)
        self.scan_button.setToolTip(tr("Rotate a scanned sheet 180° and separate every portrait"))
        self.scanners_button = button(tr("Scanners…"), self.open_scanner)
        header.addWidget(self.scanners_button)
        header.addWidget(self.scan_button)
        header.addWidget(self.open_button)
        header.addWidget(self.folder_button)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(16)
        layout.addWidget(splitter, 1)
        queue_panel = QFrame()
        queue_panel.setObjectName("card")
        queue_layout = QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(14, 18, 14, 16)
        queue_layout.setSpacing(12)
        self.queue_heading = QLabel(tr("YOUR PHOTOS  ·  {count}", count=0))
        self.queue_heading.setObjectName("section")
        queue_layout.addWidget(self.queue_heading)
        self.queue = QListWidget()
        self.queue.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.queue.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.queue.currentRowChanged.connect(self._select_photo)
        queue_layout.addWidget(self.queue, 1)
        self.batch_button = button(tr("Process next 5"), self._process_all)
        self.batch_button.setToolTip(
            tr("Prepare up to five portraits, one at a time. Nothing is saved automatically.")
        )
        queue_layout.addWidget(self.batch_button)
        self.queue_hint = hint(
            tr("Select a photo to review. Save & next moves through your folder.")
        )
        queue_layout.addWidget(self.queue_hint)
        queue_panel.setMinimumWidth(180)
        queue_panel.setMaximumWidth(240)
        splitter.addWidget(queue_panel)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(12)
        toolbar = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(
            [tr("Final portrait"), tr("Before trimming"), tr("Original & mask")]
        )
        self.view_mode.currentIndexChanged.connect(self._view_changed)
        toolbar.addWidget(self.view_mode)
        self.guide = QCheckBox(tr("Frame guides"))
        self.guide.setChecked(True)
        self.guide.toggled.connect(self._show_preview)
        toolbar.addWidget(self.guide)
        self.snap = QCheckBox(tr("Snap"))
        self.snap.setChecked(bool_setting(self.preferences, "canvas/snap", True))
        self.snap.setToolTip(
            tr("Snap to center and frame edges. Hold Alt while dragging to bypass.")
        )
        self.snap.toggled.connect(lambda checked: self.preferences.setValue("canvas/snap", checked))
        toolbar.addWidget(self.snap)
        toolbar.addStretch()
        self.canvas = Canvas()
        self.canvas.setToolTip(
            tr("Drag to position the portrait. Hold Space to pan the view. Scroll to zoom.")
        )
        zoom_out = button("−", lambda: self.canvas.zoom(1 / 1.25))
        zoom_out.setFixedWidth(36)
        zoom_out.setToolTip(tr("Zoom out"))
        toolbar.addWidget(zoom_out)
        zoom_in = button("+", lambda: self.canvas.zoom(1.25))
        zoom_in.setFixedWidth(36)
        zoom_in.setToolTip(tr("Zoom in"))
        toolbar.addWidget(zoom_in)
        toolbar.addWidget(button(tr("Fit"), self.canvas.fit))
        center_layout.addLayout(toolbar)
        center_layout.addWidget(self.canvas, 1)
        self.measurements = hint(tr("Preview only · Frame guides are hidden when you save."))
        center_layout.addWidget(self.measurements)
        splitter.addWidget(center)
        self.canvas.point_selected.connect(self._set_point)
        self.canvas.stroke_started.connect(self._begin_stroke)
        self.canvas.brush_moved.connect(self._paint)
        self.canvas.stroke_finished.connect(self._end_stroke)
        self.canvas.portrait_drag_started.connect(self._begin_position)
        self.canvas.portrait_moved.connect(self._move_position)
        self.canvas.portrait_drag_finished.connect(self._end_position)

        scroll = QScrollArea()
        self.inspector_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(316)
        scroll.setMaximumWidth(340)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 8, 0)
        controls_layout.setSpacing(14)
        scroll.setWidget(controls)
        splitter.addWidget(scroll)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([190, 800, 328])
        splitter.setStretchFactor(1, 1)

        ai_group, ai_layout = card(tr("01   Automatic setup"))
        self.ai_mode = QComboBox()
        self.ai_mode.addItem(tr("Auto · best level for your time per photo"), AUTO)
        for item in LEVELS:
            label = (
                f"{item.number} · {display_name(item)}"
                if item.bundled
                else tr(
                    "{number} · {name} · downloads ~970 MB first",
                    number=item.number,
                    name=display_name(item),
                )
            )
            self.ai_mode.addItem(label, item.number)
        stored = level_from_preference(self.preferences.value("ai/level", "auto"))
        self.ai_mode.setCurrentIndex(max(0, self.ai_mode.findData(stored or "auto")))
        self.ai_mode.setToolTip(ai_mode_tip())
        ai_layout.addWidget(self.ai_mode)
        self.auto_box = QWidget()
        auto_layout = QHBoxLayout(self.auto_box)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.setSpacing(8)
        self.ai_budget = QSpinBox()
        self.ai_budget.setRange(1, 600)
        self.ai_budget.setSuffix(tr(" s per photo"))
        self.ai_budget.setValue(int_setting(self.preferences, "ai/budget", 20))
        self.ai_budget.setToolTip(tr("How long you are willing to wait for each photo."))
        self.measure_button = button(tr("Measure this PC"), self.measure_pc)
        auto_layout.addWidget(self.ai_budget, 1)
        auto_layout.addWidget(self.measure_button)
        ai_layout.addWidget(self.auto_box)
        self.ai_estimate = hint("")
        ai_layout.addWidget(self.ai_estimate)
        self.ai_mode.currentIndexChanged.connect(self._ai_mode_changed)
        self.ai_budget.valueChanged.connect(self._ai_budget_changed)
        self._apply_ai_level()
        ai_layout.addWidget(
            hint(tr("Background removal and face alignment start when you open a photo."))
        )
        self.ai_button = button(tr("Retry AI"), self.run_ai)
        self.manual_button = button(tr("Manual setup"), self._manual)
        ai_actions = QHBoxLayout()
        ai_actions.setSpacing(8)
        ai_actions.addWidget(self.ai_button)
        ai_actions.addWidget(self.manual_button)
        ai_layout.addLayout(ai_actions)
        self.face_selector = QComboBox()
        self.face_selector.addItem(tr("No detected face"))
        self.face_selector.currentIndexChanged.connect(self._choose_face)
        ai_layout.addWidget(self.face_selector)
        self.face_selector.hide()
        controls_layout.addWidget(ai_group)

        self.place_group, place_layout = card(tr("02   Frame your portrait"))
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(12)
        form.setHorizontalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.headroom = IntegerField()
        self.headroom.setRange(0, 1047)
        self.headroom.setSuffix(PIXELS)
        self.headroom.setValue(180)
        self.headroom.valueChanged.connect(self._settings_changed)
        form.addRow(tr("Head top limit"), stepper(self.headroom))
        self.side_padding = IntegerField()
        self.side_padding.setRange(0, 473)
        self.side_padding.setSuffix(PIXELS)
        self.side_padding.setValue(64)
        self.side_padding.setToolTip(
            tr("Keep the portrait inside these equal left and right margins.")
        )
        self.side_padding.valueChanged.connect(self._settings_changed)
        form.addRow(tr("Side inset"), stepper(self.side_padding))
        self.bottom_gap = IntegerField()
        self.bottom_gap.setRange(0, 867)
        self.bottom_gap.setSuffix(PIXELS)
        self.bottom_gap.setValue(0)
        self.bottom_gap.setToolTip(
            tr("Distance from the portrait's lowest pixel to the canvas bottom.")
        )
        self.bottom_gap.valueChanged.connect(self._settings_changed)
        form.addRow(tr("Bottom gap"), stepper(self.bottom_gap))
        self.rotation = DecimalField()
        self.rotation.setRange(-180, 180)
        self.rotation.setDecimals(1)
        self.rotation.setSingleStep(0.1)
        self.rotation.setSuffix("°")
        self.rotation.valueChanged.connect(self._settings_changed)
        self.rotation.setToolTip(tr("Fine adjustment added to the automatic eye-level rotation."))
        place_layout.addLayout(form)
        self.landmark_status = hint(tr("Set nose and both eyes to enable placement."))
        self.headroom.setToolTip(
            tr(
                "Minimum distance from the top edge to the head. "
                "The portrait is anchored at the bottom."
            )
        )
        self.frame_hint = hint(tr("Bottom aligned · Natural hair preserved."))
        place_layout.addWidget(self.frame_hint)
        controls_layout.addWidget(self.place_group)

        self.retouch_toggle = QPushButton(tr("▸  Retouch && alignment"))
        self.retouch_toggle.setObjectName("disclosure")
        self.retouch_toggle.setCheckable(True)
        self.retouch_toggle.toggled.connect(self._toggle_retouch)
        controls_layout.addWidget(self.retouch_toggle)
        self.mask_group, mask_layout = card(tr("Manual corrections"))
        title_item = mask_layout.takeAt(0)
        title = title_item.widget() if title_item is not None else None
        assert title is not None
        retouch_header = QHBoxLayout()
        retouch_header.addWidget(title, 1)
        retouch_header.addWidget(button(tr("Done"), lambda: self.retouch_toggle.setChecked(False)))
        mask_layout.insertLayout(0, retouch_header)
        rotation_form = QFormLayout()
        rotation_form.setContentsMargins(0, 0, 0, 0)
        rotation_form.addRow(tr("Fine rotation"), stepper(self.rotation))
        self.horizontal_position = IntegerField()
        self.horizontal_position.setRange(-1896, 1896)
        self.horizontal_position.setSuffix(PIXELS)
        self.horizontal_position.setSingleStep(1)
        self.horizontal_position.setToolTip(
            tr("Move the portrait left or right for visual balance. Positive values move right.")
        )
        self.horizontal_position.valueChanged.connect(self._settings_changed)
        rotation_form.addRow(tr("Left / right"), stepper(self.horizontal_position))
        self.vertical_position = IntegerField()
        self.vertical_position.setRange(-2096, 2096)
        self.vertical_position.setSuffix(PIXELS)
        self.vertical_position.valueChanged.connect(self._settings_changed)
        rotation_form.addRow(tr("Up / down"), stepper(self.vertical_position))
        mask_layout.addLayout(rotation_form)
        mask_layout.addWidget(button(tr("Reset position"), self._reset_position))
        mask_layout.addWidget(self.landmark_status)
        self.strict_symmetry = QCheckBox(tr("Strict outline symmetry"))
        self.strict_symmetry.setToolTip(
            tr(
                "Trims unmatched pixels from both sides. This can cut away natural hair or ears. "
                "Leave off to preserve the complete background-removal mask."
            )
        )
        self.strict_symmetry.toggled.connect(self._settings_changed)
        mask_layout.addWidget(self.strict_symmetry)
        self.tool_group = QButtonGroup(self)
        self.tools = {}
        for row_tools in [
            [("pan", tr("Move / pan")), ("nose", tr("Nose"))],
            [("eye_a", tr("Eye 1")), ("eye_b", tr("Eye 2"))],
            [("erase", tr("Erase")), ("restore", tr("Restore"))],
        ]:
            row = QHBoxLayout()
            for tool, text in row_tools:
                b = QPushButton(text)
                b.setCheckable(True)
                b.clicked.connect(lambda checked=False, t=tool: self._tool_changed(t))
                self.tool_group.addButton(b)
                self.tools[tool] = b
                row.addWidget(b)
            mask_layout.addLayout(row)
        self.tools["pan"].setChecked(True)
        self.canvas.set_tool("pan")
        self.tool_hint = hint(tr("Choose a marker or brush to edit on the source image."))
        mask_layout.addWidget(self.tool_hint)
        brush_form = QFormLayout()
        self.brush_size = IntegerField()
        self.brush_size.setRange(2, 1000)
        self.brush_size.setValue(40)
        self.brush_size.setSuffix(PIXELS)
        self.brush_size.setToolTip(tr("Brush diameter in source-image pixels."))
        brush_form.addRow(tr("Brush size"), stepper(self.brush_size))
        self.speck_size = IntegerField()
        self.speck_size.setRange(0, 100000)
        self.speck_size.setValue(24)
        self.speck_size.setSuffix(PIXELS)
        self.speck_size.setToolTip(
            tr("Maximum area of disconnected particles to erase, in source pixels.")
        )
        brush_form.addRow(tr("Particle area ≤"), stepper(self.speck_size))
        mask_layout.addLayout(brush_form)
        self.cleanup_button = button(tr("Remove small particles"), self._cleanup)
        mask_layout.addWidget(self.cleanup_button)
        row = QHBoxLayout()
        self.undo_button = button(tr("Undo"), self._undo)
        self.redo_button = button(tr("Redo"), self._redo)
        row.addWidget(self.undo_button)
        row.addWidget(self.redo_button)
        mask_layout.addLayout(row)
        self.show_mask = QCheckBox(tr("Inspect mask in black and white"))
        self.show_mask.toggled.connect(self._inspect_mask)
        mask_layout.addWidget(self.show_mask)
        self.reset_button = button(tr("Reset this photo"), self._reset)
        mask_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.mask_group)
        self.mask_group.hide()
        controls_layout.addWidget(
            hint(tr("Natural hair is preserved. Strict trimming is optional."))
        )
        controls_layout.addStretch()
        language_row = QHBoxLayout()
        language_row.setSpacing(8)
        language_label = QLabel(tr("Language"))
        language_label.setObjectName("muted")
        self.language = QComboBox()
        for code, name in LANGUAGE_CHOICES:
            self.language.addItem(name or tr("Auto · follow Windows"), code)
        stored_language = str(self.preferences.value("ui/language", "auto"))
        self.language.setCurrentIndex(max(0, self.language.findData(stored_language)))
        self.language.currentIndexChanged.connect(self._language_changed)
        language_label.setBuddy(self.language)
        language_row.addWidget(language_label)
        language_row.addWidget(self.language, 1)
        controls_layout.addLayout(language_row)

        self.status = QLabel(
            tr("Add photos to get started. Background removal runs automatically.")
        )
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        export_card = QFrame()
        export_card.setObjectName("card")
        export_layout = QVBoxLayout(export_card)
        export_layout.setContentsMargins(18, 14, 18, 14)
        export_layout.setSpacing(10)
        filename_row = QHBoxLayout()
        filename_row.setSpacing(10)
        filename_row.addWidget(QLabel(tr("Save as")))
        self.export_name = QLineEdit()
        self.export_name.setPlaceholderText(tr("Give your photo a name"))
        self.export_name.setMinimumWidth(180)
        self.export_name.setClearButtonEnabled(True)
        self.export_name.textChanged.connect(self._name_changed)
        filename_row.addWidget(self.export_name, 1)
        self.export_format = QComboBox()
        for title, value in [
            ("PNG", "PNG"),
            ("JPG / JPEG", "JPEG"),
            ("WebP", "WEBP"),
            ("TIFF", "TIFF"),
            ("BMP", "BMP"),
        ]:
            self.export_format.addItem(title, value)
        self.export_format.currentIndexChanged.connect(self._export_options_changed)
        self.export_format.setToolTip(tr("Choose the file format for this photo."))
        filename_row.addWidget(self.export_format)
        self.quality_label = QLabel(tr("Quality"))
        self.export_quality = IntegerField()
        self.export_quality.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.export_quality.setRange(1, 100)
        self.export_quality.setValue(95)
        self.export_quality.setSuffix("%")
        self.export_quality.setFixedWidth(72)
        self.export_quality.valueChanged.connect(self._export_options_changed)
        filename_row.addWidget(self.quality_label)
        filename_row.addWidget(self.export_quality)
        self.quality_label.hide()
        self.export_quality.hide()
        filename_row.addSpacing(12)
        self.cancel_button = button(tr("Cancel"), self._cancel)
        self.cancel_button.hide()
        filename_row.addWidget(self.cancel_button)
        self.save_button = button(tr("Save photo"), self._save)
        self.next_button = button(tr("Save && next  →"), lambda: self._save(True), primary=True)
        self.next_button.setToolTip(tr("Save this photo and open the next one (Enter)"))
        filename_row.addWidget(self.save_button)
        filename_row.addWidget(self.next_button)
        export_layout.addLayout(filename_row)
        destination = QHBoxLayout()
        destination.setSpacing(16)
        self.output_label = hint(tr("Your saved photos will appear in processed_photos."))
        self.output_label.setWordWrap(False)
        self.output_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        destination.addWidget(self.output_label, 1)
        self.name_hint = hint(tr("{format} · White background", format="PNG"))
        destination.addWidget(self.name_hint)
        self.destination_button = button(tr("Change folder…"), self.choose_output)
        destination.addWidget(self.destination_button)
        export_layout.addLayout(destination)
        layout.addWidget(export_card)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)

    def _toggle_retouch(self, opened):
        self.mask_group.setVisible(opened)
        self.retouch_toggle.setText(
            tr("▾  Retouch && alignment") if opened else tr("▸  Retouch && alignment")
        )

        if opened:
            QTimer.singleShot(
                0, lambda: self.inspector_scroll.verticalScrollBar().setValue(self.mask_group.y())
            )
        else:
            self.inspector_scroll.verticalScrollBar().setValue(0)

    def _language_changed(self):
        self.preferences.setValue("ui/language", self.language.currentData())
        self.status.setText(tr("Language changes apply the next time you open Portrait Processor."))

    def _shortcuts(self):
        for key, function in [
            (QKeySequence.StandardKey.Open, self.open_photos),
            (QKeySequence.StandardKey.Save, self._save),
            (QKeySequence.StandardKey.Undo, self._undo),
            (QKeySequence.StandardKey.Redo, self._redo),
        ]:
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(function)
            self.addAction(action)
        # Enter (main or keypad) is Save & next, even while typing a name or a value.
        action = QAction(self)
        action.setShortcuts([QKeySequence(Qt.Key.Key_Return), QKeySequence(Qt.Key.Key_Enter)])
        action.triggered.connect(self._save_and_next_shortcut)
        self.addAction(action)

    def _save_and_next_shortcut(self):
        focused = QApplication.focusWidget()
        if isinstance(focused, QAbstractSpinBox):
            focused.interpretText()  # apply a value still being typed before saving
        if self.next_button.isEnabled():
            self.next_button.click()

    def _submit(self, function, done, failed, kind):
        self.serial += 1
        task = Task(self.serial, function)
        task.signals.progress.connect(self._task_progress)
        task.signals.complete.connect(self._task_complete)
        self.tasks[self.serial] = (task, done, failed, kind)
        (self.preparation_pool if kind == "batch" else self.pool).start(task)
        return self.serial

    @Slot(object)
    def _task_progress(self, payload):
        identifier, message = payload
        if identifier == self.analysis_job:
            self.status.setText(message)
        elif identifier == self.batch_job and self.batch_state is not None:
            if isinstance(message, dict):
                self._commit_batch_item(message["ready"], self.batch_state)
            else:
                self.batch_state["message"] = message
                self._update_batch_hint()
                if self.waiting_path is not None:
                    self.status.setText(
                        tr(
                            "Waiting for {name} · {message}",
                            name=self.waiting_path.name,
                            message=message,
                        )
                    )

    @Slot(object)
    def _task_complete(self, payload):
        identifier, result, error = payload
        entry = self.tasks.pop(identifier, None)
        if entry is None:
            return
        _, done, failed, _ = entry
        if identifier == self.analysis_job:
            self.analysis_job = None
            self._busy(False)
        if identifier == self.batch_job:
            self.batch_job = None
        if error is None:
            done(result)
        else:
            failed(error)
        if self.pending_close and not self.tasks:
            self.pending_close = False
            self.close()

    def measuring(self):
        return self.measure_job is not None and self.measure_job in self.tasks

    def _busy(self, busy):
        # A running measurement keeps the window busy whatever photo selection asks for.
        busy = busy or self.measuring()
        if busy and not self.analysis_busy:
            self.busy_started = time.monotonic()
            self.elapsed_text = None
            self.elapsed_timer.start()
        elif not busy:
            self.elapsed_timer.stop()
        self.analysis_busy = busy
        can_open = not busy and self.batch_job is None
        for control in (
            self.open_button,
            self.folder_button,
            self.scan_button,
            self.scanners_button,
        ):
            control.setEnabled(can_open)
            control.setToolTip(
                tr("Pause preparation to open another queue.") if self.batch_job else ""
            )
        self.queue.setEnabled(not busy or self.waiting_path is not None)
        self.cancel_button.setVisible(busy or self.batch_job is not None)
        self.cancel_button.setText(
            tr("Pause preparation") if self.batch_job and not busy else tr("Cancel")
        )
        self.progress.setVisible(busy or self.batch_job is not None)
        self._refresh_enabled()

    def _update_elapsed(self):
        if self.status.text() != self.elapsed_text:
            self.operation_message = self.status.text()
        seconds = int(time.monotonic() - self.busy_started)
        self.elapsed_text = tr(
            "{message} · {minutes}:{seconds:02d} elapsed",
            message=self.operation_message,
            minutes=seconds // 60,
            seconds=seconds % 60,
        )
        self.status.setText(self.elapsed_text)

    def _refresh_enabled(self):
        editable = self.doc is not None and not self.analysis_busy
        self.export_name.setEnabled(editable)
        self.export_format.setEnabled(editable)
        self.export_quality.setEnabled(editable)
        self.batch_button.setEnabled(
            bool(self.paths)
            and not self.analysis_busy
            and self.batch_job is None
            and len(self.prepared) < len(self.paths)
        )
        self.batch_button.setText(
            tr("Preparing…")
            if self.batch_job
            else tr("Resume preparation")
            if self.prefetch_paused
            else tr("Process next 5")
        )
        valid_name = False
        if self.doc is not None:
            try:
                export_stem(self.export_name.text())
                valid_name = True
            except ValueError:
                pass
        self.place_group.setEnabled(editable)
        self.mask_group.setEnabled(editable)
        self.canvas.setEnabled(not self.analysis_busy)
        self.ai_button.setEnabled(editable and self.batch_job is None)
        idle = not self.analysis_busy and self.batch_job is None
        for control in (self.ai_mode, self.ai_budget, self.measure_button):
            control.setEnabled(idle)
        self.ai_button.setToolTip(
            tr("Pause background preparation before retrying AI.") if self.batch_job else ""
        )
        self.ai_mode.setToolTip(
            tr("Pause background preparation to change accuracy.")
            if self.batch_job
            else ai_mode_tip()
        )
        self.manual_button.setEnabled(editable)
        self.face_selector.setEnabled(editable and bool(self.doc.faces) if self.doc else False)
        current = (
            self.doc is not None
            and editable
            and valid_name
            and self.stroke_before is None
            and self.drag_state is None
            and self.doc.ready
            and self.doc.result is not None
            and self.doc.result_revision == self.doc.revision
            and bool(np.any(self.doc.result["alpha"]))
        )
        self.save_button.setEnabled(bool(current))
        self.next_button.setEnabled(bool(current))
        self.undo_button.setEnabled(editable and bool(self.doc.undo_stack) if self.doc else False)
        self.redo_button.setEnabled(editable and bool(self.doc.redo_stack) if self.doc else False)

    def _confirm_discard(self):
        unsaved = sum(doc.dirty for doc in self.documents.values()) + sum(self.draft_dirty.values())
        if not unsaved:
            return True
        answer = QMessageBox.question(
            self,
            tr("Unsaved edits"),
            tr("Discard unsaved edits for {count} photo(s)?", count=unsaved),
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def open_photos(self):
        if self.analysis_busy or self.batch_job is not None:
            return
        names, _ = QFileDialog.getOpenFileNames(
            self,
            tr("Open portraits"),
            str(Path.home() / "Pictures"),
            tr("Photos ({patterns})", patterns=IMAGE_PATTERNS),
        )
        if names:
            self.open_paths([Path(name) for name in names])

    def open_folder(self):
        if self.analysis_busy or self.batch_job is not None:
            return
        name = QFileDialog.getExistingDirectory(
            self, tr("Open photo folder"), str(Path.home() / "Pictures")
        )
        if name:
            paths = sorted(
                (p for p in Path(name).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED),
                key=lambda p: p.name.casefold(),
            )
            if not paths:
                QMessageBox.information(
                    self, tr("No photos"), tr("This folder contains no supported photos.")
                )
                return
            self.open_paths(paths)

    def open_scan(self):
        if self.analysis_busy or self.batch_job is not None:
            return
        name, _ = QFileDialog.getOpenFileName(
            self,
            tr("Import a scanned sheet"),
            str(Path.home() / "Pictures"),
            tr("Scans ({patterns})", patterns=IMAGE_PATTERNS),
        )
        if name:
            self.open_scan_path(Path(name))

    def open_scanner(self):
        if self.analysis_busy or self.batch_job is not None:
            return
        from .scanner_dialog import ScannerDialog

        self.timer.stop()
        self.engine.close()  # Free model memory while acquiring a high-resolution sheet.
        dialog = ScannerDialog(self, preferences=self.preferences)
        if dialog.exec() == ScannerDialog.DialogCode.Accepted and dialog.accepted_path:
            self.open_scan_path(dialog.accepted_path)
        else:
            self._schedule_render()
        dialog.deleteLater()

    def open_scan_path(self, path, *, output=None):
        if self.analysis_busy or self.batch_job is not None:
            return
        self.timer.stop()
        self._busy(True)
        self.status.setText(tr("Opening scanned sheet…"))

        def review(scan):
            dialog = ScanReview(scan, self)
            accepted = dialog.exec() == ScanReview.DialogCode.Accepted
            crops = dialog.canvas.crops
            dialog.timer.stop()
            dialog.deleteLater()
            if not accepted:
                self.status.setText(
                    tr("Scan import cancelled. Your current photos are still open.")
                )
                self._schedule_render()
                return
            self._import_scan(scan, crops, output=output)

        self.analysis_job = self._submit(
            lambda progress, cancel: load_scan(path, progress, cancel), review, self._failed, "scan"
        )

    def _import_scan(self, scan, crops, *, output=None):
        if (
            not crops
            or self.analysis_busy
            or self.batch_job is not None
            or not self._confirm_discard()
        ):
            return
        directory = tempfile.TemporaryDirectory(prefix="portrait-scan-")
        self._busy(True)

        def extracted(paths):
            self.open_paths(
                paths,
                output=output if output is not None else scan.path.parent / "processed_photos",
                scan_directory=directory,
                auto_analyze=False,
                prepare_all=True,
                batch_mode=True,
                confirmed=True,
            )

        def failed(error):
            directory.cleanup()
            self._failed(error)

        self.analysis_job = self._submit(
            lambda progress, cancel: extract_scan(scan, crops, directory.name, progress, cancel),
            extracted,
            failed,
            "scan-extract",
        )

    def open_paths(
        self,
        paths,
        *,
        output=None,
        scan_directory=None,
        prepare_all=False,
        confirmed=False,
        auto_analyze=True,
        batch_mode=False,
    ):
        if (
            self.analysis_busy
            or self.batch_job is not None
            or not paths
            or (not confirmed and not self._confirm_discard())
        ):
            return
        self.timer.stop()
        self.doc = None
        self.auto_analyze = auto_analyze
        self.batch_mode = batch_mode
        self.prefetch_enabled = batch_mode or prepare_all
        self.prefetch_paused = False
        self.waiting_path = None
        self.last_review_path = None
        self.batch_state = None
        self.prepared.clear()
        self.manual_required.clear()
        self.documents.clear()
        self.drafts.clear()
        self.draft_dirty.clear()
        self.session_directory.cleanup()
        self.session_directory = tempfile.TemporaryDirectory(prefix="portrait-session-")
        if self.scan_directory is not None:
            self.scan_directory.cleanup()
        self.scan_directory = scan_directory
        self.paths = list(dict.fromkeys(p.resolve() for p in paths))
        self.output = output if output is not None else self.paths[0].parent / "processed_photos"
        self._update_output()
        self.queue.blockSignals(True)
        self.queue.clear()
        self.queue.addItems([p.name for p in self.paths])
        for index, path in enumerate(self.paths):
            self.queue.item(index).setToolTip(path.name)
        self.queue_heading.setText(tr("YOUR PHOTOS  ·  {count}", count=len(self.paths)))
        self.queue_hint.setText(
            tr("Imported only. Drag to position. Process photo or Process next 5 starts AI.")
            if not auto_analyze
            else tr("Select a photo to review. Save & next moves through your folder.")
        )
        if prepare_all:
            self.queue.setCurrentRow(0)
        self.queue.blockSignals(False)
        if prepare_all:
            self._process_all()
            return
        self.queue.setCurrentRow(0)

    def _process_all(self, checked=False, *, focus_index=None, automatic=False):
        if self.batch_job is not None or self.analysis_job is not None or not self.paths:
            return
        if self.analysis_busy and self.waiting_path is None:
            return
        if automatic and self.prefetch_paused:
            return
        self.prefetch_enabled = True
        self.prefetch_paused = False
        if focus_index is None and self.waiting_path is not None:
            focus_index = self.paths.index(self.waiting_path)
        candidates = (
            self.paths
            if focus_index is None
            else self.paths[focus_index // 5 * 5 : focus_index // 5 * 5 + 5]
        )
        candidates = [p for p in candidates if p not in self.prepared][:5]
        if not candidates:
            self.status.setText(
                tr("These portraits are already prepared. Review, name, and save them.")
            )
            return
        selected = max(0, self.queue.currentRow()) if focus_index is None else focus_index
        interactive = (
            self.doc is not None and self.doc.path not in candidates and not self.analysis_busy
        )
        if not interactive:
            self.timer.stop()
            self.waiting_path = self.paths[selected]
            self.doc = None
            self.queue.blockSignals(True)
            self.queue.setCurrentRow(selected)
            self.queue.blockSignals(False)
            self.status.setText(
                tr("Preparing {count} portraits · one at a time…", count=len(candidates))
            )
        state = {
            "committed": set(),
            "initial": not self.prepared,
            "message": tr("Preparing the next portraits…"),
        }
        self.batch_state = state
        completed = []
        defaults: dict[str, Any] = {
            "headroom": int_setting(self.preferences, "frame/headroom", 180),
            "side_margin": int_setting(self.preferences, "frame/side_margin", 64),
            "bottom_margin": int_setting(self.preferences, "frame/bottom_margin", 0),
            "strict_symmetry": bool_setting(self.preferences, "frame/strict_symmetry", False),
        }
        image_format = str(self.preferences.value("export/format", "PNG"))
        quality = min(100, max(1, int_setting(self.preferences, "export/quality", 95)))
        batch_directory = Path(self.session_directory.name) / f"batch-{self.serial + 1}"
        batch_directory.mkdir()
        sources = [(p, self.documents.get(p), self.drafts.get(p)) for p in candidates]

        def prepare_items(progress, cancel):
            for index, (path, cached, draft) in enumerate(sources):
                check_cancel(cancel)
                progress(
                    tr(
                        "Portrait {number}/{total} · {name}",
                        number=index + 1,
                        total=len(sources),
                        name=path.name,
                    )
                )
                if cached is not None:
                    doc = copy.copy(cached)
                    doc.settings = replace(cached.settings)
                elif draft is not None:
                    doc = Document.from_stash(path, draft)
                else:
                    doc = Document.load(path, **defaults)
                    doc.export_format = image_format if image_format in EXPORT_FORMATS else "PNG"
                    doc.export_quality = quality
                if not doc.analyzed:
                    alpha, faces, _warnings = self.engine.analyze(
                        doc.rgb,
                        doc.original_alpha,
                        lambda message, index=index: progress(
                            tr(
                                "Portrait {number}/{total} · {name}",
                                number=index + 1,
                                total=len(sources),
                                name=message,
                            )
                        ),
                        cancel,
                    )
                    check_cancel(cancel)
                    doc.apply_analysis(alpha, faces)
                # Validate placement now; previews are rebuilt cheaply when reviewed.
                if doc.ready and doc.result_revision != doc.revision:
                    doc.result = render_document(doc)
                    doc.result_revision = doc.revision
                check_cancel(cancel)
                target = batch_directory / f"{index}.npz"
                doc.stash(target)
                item = (path, target, doc.dirty, doc.ready, doc.saved_path)
                completed.append(item)
                progress({"ready": item})

        def prepare(progress, cancel):
            try:
                return prepare_items(progress, cancel)
            finally:
                self.engine.close()  # Release model allocations between groups.

        def finished(error=None):
            for item in completed:
                self._commit_batch_item(item, state)
            state["initial"] = False
            self.batch_state = None
            if self.pending_close:
                self.waiting_path = None
                self._busy(False)
                self._update_batch_hint()
                return
            if error is not None:
                self.prefetch_paused = True
            if self.waiting_path is not None:
                path = self.waiting_path
                if path in self.prepared:
                    self.waiting_path = None
                    self._busy(False)
                    self._select_photo(self.paths.index(path), analyze=False)
                elif error is None:
                    self._process_all(focus_index=self.paths.index(path), automatic=True)
                else:
                    self.waiting_path = None
                    self._busy(False)
                    fallback = self.last_review_path or next(
                        (p for p in self.paths if p in self.prepared), None
                    )
                    if fallback is not None:
                        self.queue.blockSignals(True)
                        self.queue.setCurrentRow(self.paths.index(fallback))
                        self.queue.blockSignals(False)
                        self._select_photo(self.paths.index(fallback), analyze=False)
            else:
                self._busy(self.analysis_busy)
                self._start_prefetch()
            if error is not None:
                self.status.setText(
                    tr("Preparation paused. Completed portraits and edits are kept.")
                    if isinstance(error, Cancelled)
                    else str(error)
                )
            self._update_batch_hint()

        self.batch_job = self._submit(prepare, lambda _: finished(), finished, "batch")
        self._busy(not interactive)
        self._update_batch_hint()

    def _commit_batch_item(self, item, state):
        path, draft, dirty, ready, saved = item
        if path in state["committed"]:
            return
        state["committed"].add(path)
        self.prepared.add(path)
        if not ready:
            self.manual_required.add(path)
        self.documents.pop(path, None)
        self.drafts[path] = draft
        self.draft_dirty[path] = dirty
        prefix = "✓ " if saved and not dirty else "● " if ready else "! "
        self.queue.item(self.paths.index(path)).setText(prefix + path.name)
        if self.waiting_path == path and not state["initial"]:
            self.waiting_path = None
            self._busy(False)
            self._select_photo(self.paths.index(path), analyze=False)
        self._update_batch_hint()

    def _start_prefetch(self):
        if (
            self.prefetch_enabled
            and not self.prefetch_paused
            and not self.pending_close
            and not self.analysis_busy
            and self.analysis_job is None
            and self.batch_job is None
            and self.doc is not None
            and len(self.prepared) < len(self.paths)
        ):
            self._process_all(automatic=True)

    def _update_batch_hint(self):
        if self.batch_job is not None and self.batch_state is not None:
            state = tr("Preparing more in the background. Ready portraits stay editable.")
            self.queue_hint.setToolTip(str(self.batch_state["message"]))
        elif self.prefetch_paused:
            state = tr("Preparation paused. Resume when ready.")
        else:
            state = tr("Review, name, then Save & next.")
        summary = tr(
            "{done}/{total} processed. {state}",
            done=len(self.prepared),
            total=len(self.paths),
            state=state,
        )
        if self.manual_required:
            summary += " " + tr("{count} need manual markers.", count=len(self.manual_required))
        self.queue_hint.setText(summary)
        if self.batch_job is None:
            self.queue_hint.setToolTip(summary)
        self._refresh_enabled()

    def _select_photo(self, index, *, analyze=None):
        if (
            (self.analysis_busy and self.waiting_path is None)
            or self.measuring()
            or not 0 <= index < len(self.paths)
        ):
            return
        path = self.paths[index]
        if (
            (self.batch_mode or self.prefetch_enabled)
            and analyze is None
            and path not in self.prepared
        ):
            self.waiting_path = path
            self.timer.stop()
            self.doc = None
            self._busy(not self.prefetch_paused or self.batch_job is not None)
            self.status.setText(
                tr("Waiting for {name}…", name=path.name)
                if not self.prefetch_paused
                else tr("Preparation paused. Resume to prepare this portrait.")
            )
            if self.batch_job is None and not self.prefetch_paused:
                self._process_all(focus_index=index, automatic=True)
            return
        self.waiting_path = None
        self.timer.stop()
        self.doc = None
        self._refresh_enabled()
        path = self.paths[index]
        analyze = self.auto_analyze if analyze is None else analyze
        if path in self.documents:
            self.documents[path] = self.documents.pop(path)
            self._activate(self.documents[path])
            self._busy(False)
            self._start_prefetch()
            return
        self._busy(True)
        self.status.setText(tr("Opening {name}…", name=path.name))
        headroom = int_setting(self.preferences, "frame/headroom", 180)
        side_margin = int_setting(self.preferences, "frame/side_margin", 64)
        bottom_margin = int_setting(self.preferences, "frame/bottom_margin", 0)
        strict_symmetry = bool_setting(self.preferences, "frame/strict_symmetry", False)
        export_format = str(self.preferences.value("export/format", "PNG"))
        export_quality = int_setting(self.preferences, "export/quality", 95)
        victims = list(self.documents.items())[: max(0, len(self.documents) - 2)]

        def load(progress, cancel):
            stashed = []
            for old_path, old_doc in victims:
                check_cancel(cancel)
                draft = Path(self.session_directory.name) / f"{self.paths.index(old_path)}.npz"
                old_doc.stash(draft)
                stashed.append((old_path, draft, old_doc.dirty))
            check_cancel(cancel)
            doc = (
                Document.from_stash(path, self.drafts[path])
                if path in self.drafts
                else Document.load(
                    path,
                    headroom,
                    side_margin=side_margin,
                    bottom_margin=bottom_margin,
                    strict_symmetry=strict_symmetry,
                )
            )
            if path not in self.drafts:
                doc.export_format = export_format if export_format in EXPORT_FORMATS else "PNG"
                doc.export_quality = min(100, max(1, export_quality))
                if not analyze:
                    doc.manual_placement = True
            return doc, stashed

        def loaded(payload):
            doc, stashed = payload
            for old_path, draft, dirty in stashed:
                self.documents.pop(old_path, None)
                self.drafts[old_path] = draft
                self.draft_dirty[old_path] = dirty
            self.drafts.pop(path, None)
            self.draft_dirty.pop(path, None)
            self._loaded(doc, analyze=analyze)

        self.analysis_job = self._submit(load, loaded, self._failed, "load")

    def _loaded(self, doc, *, analyze=True):
        self.documents[doc.path] = doc
        self._activate(doc)
        if analyze and not doc.analyzed:
            self._run_analysis()
        else:
            self._start_prefetch()

    def _activate(self, doc):
        self.doc = doc
        self.last_review_path = doc.path
        self.syncing = True
        self.headroom.setMaximum(1047)
        self.bottom_gap.setMaximum(1047)
        self.headroom.setValue(doc.settings.headroom)
        self.side_padding.setValue(doc.settings.side_margin)
        self.bottom_gap.setValue(doc.settings.bottom_margin)
        self.horizontal_position.setValue(doc.settings.position_x)
        self.vertical_position.setValue(doc.settings.position_y)
        self.strict_symmetry.setChecked(doc.settings.strict_symmetry)
        self.frame_hint.setText(
            tr("Bottom aligned · Strict outline trimming.")
            if doc.settings.strict_symmetry
            else tr("Bottom aligned · Natural hair preserved.")
        )
        self.headroom.setMaximum(1047 - self.bottom_gap.value())
        self.bottom_gap.setMaximum(1047 - self.headroom.value())
        self.export_name.setText(doc.output_name if doc.output_name is not None else doc.path.stem)
        self.export_format.setCurrentIndex(max(0, self.export_format.findData(doc.export_format)))
        self.export_quality.setValue(doc.export_quality)
        self.quality_label.setVisible(doc.export_format in {"JPEG", "WEBP"})
        self.export_quality.setVisible(doc.export_format in {"JPEG", "WEBP"})
        self._name_hint()
        self.rotation.setValue(doc.settings.angle_offset)
        self.face_selector.setVisible(len(doc.faces) > 1)
        self.face_selector.clear()
        self.face_selector.addItem(
            tr("Select detected face…") if doc.faces else tr("Set markers manually")
        )
        for i, face in enumerate(doc.faces):
            self.face_selector.addItem(
                tr("Face {number} · nose at x={x:.0f}", number=i + 1, x=face.nose[0])
            )
        selected = next((i + 1 for i, face in enumerate(doc.faces) if face.nose == doc.nose), 0)
        self.face_selector.setCurrentIndex(selected)
        self.syncing = False
        self.ai_button.setText(tr("Retry AI") if doc.analyzed else tr("Process photo"))
        self._tool_changed("pan")
        self.view_mode.setCurrentIndex(0 if doc.ready else 2)
        self._show_preview()
        self._landmark_label()
        self.status.setText(
            tr("Reviewing {name}", name=doc.path.name)
            if doc.ready
            else tr("Set nose and both eyes on the source image.")
        )
        self._refresh_enabled()
        if doc.ready and doc.result_revision != doc.revision:
            self._schedule_render()

    def run_ai(self):
        if self.doc is None or self.analysis_busy or self.batch_job is not None:
            return
        if (
            self.doc.analyzed
            and QMessageBox.question(
                self,
                tr("Run AI again"),
                tr("Replace the current mask and landmarks with a fresh AI result?"),
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._run_analysis()

    def _ai_mode_changed(self):
        if self.analysis_busy or self.batch_job is not None:
            return
        self.preferences.setValue("ai/level", str(self.ai_mode.currentData()))
        self._apply_ai_level()
        self.status.setText(
            tr("Accuracy applies to portraits processed next. Existing edits are kept.")
        )

    def _ai_budget_changed(self, value):
        self.preferences.setValue("ai/budget", value)
        self._apply_ai_level()

    def _apply_ai_level(self):
        """Choose the engine level from the fixed choice, or from the budget in Auto."""
        auto = self.ai_mode.currentData() == "auto"
        self.auto_box.setVisible(auto)
        self.ai_estimate.setVisible(auto)
        if not auto:
            self.engine.level = self.ai_mode.currentData()
            return
        measurement = load_measurement(self.preferences)
        if measurement is None:
            self.engine.level = 1
            self.ai_estimate.setText(
                tr(
                    "Not measured yet: using level 1 · Fastest. Choose Measure this PC "
                    "(about a minute) to find the best level for your time per photo."
                )
            )
            return
        chosen, seconds, fits = measurement.choice(self.ai_budget.value())
        self.engine.level = chosen.number
        when = measurement.date
        predicted = chosen.number != 1 and chosen.number not in measurement.confirmed
        values = {"number": chosen.number, "name": display_name(chosen), "seconds": seconds}
        estimate = (
            tr("This PC: level {number} · {name} ≈ {seconds:.0f} s per photo", **values)
            if fits
            else tr(
                "This PC: level {number} · {name} ≈ {seconds:.0f} s per photo"
                " — over your time per photo, so the fastest level is used",
                **values,
            )
        )
        self.ai_estimate.setText(
            tr(
                "{estimate} (predicted, not yet tried on this PC). Measured {date}; "
                "choose Measure this PC again to confirm.",
                estimate=estimate,
                date=when,
            )
            if predicted
            else tr(
                "{estimate}. Measured {date}; the first photo also loads the model.",
                estimate=estimate,
                date=when,
            )
        )

    def measure_pc(self):
        if (
            self.analysis_busy
            or self.batch_job is not None
            or self.waiting_path is not None
            or self.measuring()
        ):
            return
        budget = self.ai_budget.value()
        engine = self.engine
        rgb, alpha = calibration_photo()
        self._busy(True)
        self.status.setText(tr("Measuring this PC…"))

        def run(progress, cancel):
            return measure_levels(
                lambda number: engine.measure_level(number, rgb, alpha, progress, cancel),
                budget,
                progress,
                cancel,
            )

        def done(measurement):
            save_measurement(self.preferences, measurement)
            self._apply_ai_level()
            self.status.setText(self.ai_estimate.text())
            self._start_prefetch()

        def failed(error):
            self.status.setText(
                tr("Measurement cancelled. The previous result is kept.")
                if isinstance(error, Cancelled)
                else tr("Could not measure this PC: {error}", error=error)
            )
            self._refresh_enabled()
            self._start_prefetch()

        self.analysis_job = self.measure_job = self._submit(run, done, failed, "analysis")

    def _run_analysis(self):
        doc = self.doc
        if doc is None or self.batch_job is not None:
            return
        self.timer.stop()
        self._busy(True)
        self.status.setText(tr("Preparing local AI…"))

        def done(result):
            alpha, faces, warnings = result
            doc.apply_analysis(alpha, faces)
            self.prepared.add(doc.path)
            self._activate(doc)
            if warnings:
                self.status.setText(" ".join(warnings))
            elif not faces:
                self.status.setText(
                    tr("No face detected. Click Nose, Eye 1, and Eye 2 on the source image.")
                )
            elif len(faces) > 1:
                self.status.setText(
                    tr("Several faces detected. Select a face or place markers manually.")
                )
            else:
                self.view_mode.setCurrentIndex(0)
                self.status.setText(tr("Background removed. Preparing the centered preview…"))

        self.analysis_job = self._submit(
            lambda progress, cancel: self.engine.analyze(
                doc.rgb, doc.original_alpha, progress, cancel
            ),
            done,
            self._failed,
            "analysis",
        )

    def _failed(self, error):
        message = str(error)
        if not isinstance(error, Cancelled):
            if self.doc is None:
                message = tr(
                    "Could not open photo: {message}  Choose another photo.", message=message
                )
            else:
                message = tr(
                    "{message}  Retry AI, or choose Edit without AI for manual corrections.",
                    message=message,
                )
        self.status.setText(message)
        self._refresh_enabled()

    def _manual(self):
        if self.doc is None or self.analysis_busy:
            return
        self.doc.analyzed = True
        self.doc.manual_placement = True
        self.doc.changed()
        self.view_mode.setCurrentIndex(2)
        self._tool_changed("nose")
        self.status.setText(
            tr("Manual editing enabled. Place the nose and both eyes, then erase the background.")
        )
        self._refresh_enabled()
        self._schedule_render()

    def _choose_face(self, index):
        if self.syncing or self.doc is None or index < 1 or self.analysis_busy:
            return
        self.doc.choose_face(index - 1)
        self._landmark_label()
        self._show_preview()
        self._schedule_render()

    def _landmark_label(self):
        doc = self.doc
        if doc is None:
            return
        if doc.ready and doc.path in self.manual_required:
            self.manual_required.discard(doc.path)
            self._update_batch_hint()
        missing = [
            name
            for name, value in [
                (tr("nose"), doc.nose),
                (tr("eye 1"), doc.eye_a),
                (tr("eye 2"), doc.eye_b),
            ]
            if value is None
        ]
        if missing:
            self.landmark_status.setText(tr("Set: {markers}.", markers=", ".join(missing)))
        else:
            try:
                self.landmark_status.setText(
                    tr(
                        "Automatic correction: {degrees:+.1f}°",
                        degrees=doc.angle - doc.settings.angle_offset,
                    )
                )
            except ValueError as error:
                self.landmark_status.setText(str(error))

    def _settings_changed(self):
        if self.syncing or self.doc is None or self.analysis_busy:
            return
        self.doc.settings.headroom = self.headroom.value()
        self.doc.settings.side_margin = self.side_padding.value()
        self.doc.settings.bottom_margin = self.bottom_gap.value()
        self.doc.settings.position_x = self.horizontal_position.value()
        self.doc.settings.position_y = self.vertical_position.value()
        self.doc.settings.strict_symmetry = self.strict_symmetry.isChecked()
        self.frame_hint.setText(
            tr("Bottom aligned · Strict outline trimming.")
            if self.strict_symmetry.isChecked()
            else tr("Bottom aligned · Natural hair preserved.")
        )
        self.headroom.setMaximum(1047 - self.bottom_gap.value())
        self.bottom_gap.setMaximum(1047 - self.headroom.value())
        self.doc.settings.angle_offset = self.rotation.value()
        self.preferences.setValue("frame/headroom", self.headroom.value())
        self.preferences.setValue("frame/side_margin", self.side_padding.value())
        self.preferences.setValue("frame/bottom_margin", self.bottom_gap.value())
        self.preferences.setValue("frame/strict_symmetry", self.strict_symmetry.isChecked())
        self.doc.changed()
        self._schedule_render()

    def _name_hint(self):
        try:
            name = export_stem(self.export_name.text())
            image_format = self.export_format.currentData()
            self.name_hint.setText(tr("{format} · White background", format=image_format))
            self.format_badge.setText(format_badge(image_format))
            self.export_name.setToolTip(
                tr(
                    "Will save as {name}.{extension}",
                    name=name,
                    extension=EXPORT_FORMATS[image_format],
                )
            )
            self.export_name.setStyleSheet("")
        except ValueError as error:
            self.name_hint.setText(str(error))
            self.export_name.setToolTip(str(error))
            self.export_name.setStyleSheet("border: 1px solid #e39785;")

    def _name_changed(self):
        if self.syncing or self.doc is None:
            return
        self.doc.output_name = self.export_name.text()
        suffix = self.export_name.text().strip().rsplit(".", 1)
        if len(suffix) == 2 and suffix[-1].lower() in EXPORT_SUFFIXES:
            self.syncing = True
            self.export_format.setCurrentIndex(
                self.export_format.findData(EXPORT_SUFFIXES[suffix[-1].lower()])
            )
            self.syncing = False
        self._export_options_changed(normalize_name=False)
        self.doc.dirty = True
        self._name_hint()
        self._refresh_enabled()

    def _export_options_changed(self, *args, normalize_name=True):
        if self.syncing or self.doc is None:
            return
        image_format = self.export_format.currentData()
        if normalize_name:
            try:
                normalized = export_stem(self.export_name.text())
                self.syncing = True
                self.export_name.setText(normalized)
            except ValueError:
                pass
            finally:
                self.syncing = False
        self.doc.output_name = self.export_name.text()
        self.doc.export_format = image_format
        self.doc.export_quality = self.export_quality.value()
        self.doc.dirty = True
        self.quality_label.setVisible(image_format in {"JPEG", "WEBP"})
        self.export_quality.setVisible(image_format in {"JPEG", "WEBP"})
        self.preferences.setValue("export/format", image_format)
        self.preferences.setValue("export/quality", self.export_quality.value())
        self._name_hint()
        self._refresh_enabled()

    def _schedule_render(self):
        self._refresh_enabled()
        if (
            self.doc is not None
            and self.doc.ready
            and not self.analysis_busy
            and self.drag_state is None
        ):
            self.status.setText(tr("Updating preview…"))
            self.timer.start()

    def _render(self):
        if (
            self.doc is None
            or not self.doc.ready
            or self.analysis_busy
            or self.drag_state is not None
        ):
            return
        if self.render_running:
            self.timer.start()
            return
        doc = self.doc
        revision = doc.revision
        snapshot = copy.copy(doc)
        snapshot.alpha = doc.alpha.copy()
        snapshot.settings = replace(doc.settings)
        self.render_running = True

        def done(result):
            self.render_running = False
            if doc is not self.doc or doc.revision != revision:
                self._schedule_render()
                return
            doc.result = result
            doc.result_revision = revision
            self._show_preview()
            removed = result["removed_fraction"]
            if not np.any(result["alpha"]):
                self.status.setText(
                    tr(
                        "No portrait is visible. Use Reset position, correct the markers, "
                        "or restore the mask."
                    )
                )
            elif removed > 0.25:
                self.status.setText(
                    tr(
                        "Ready to review · symmetric trim removes {removed:.0%} of foreground. "
                        "Check the alignment and mask.",
                        removed=removed,
                    )
                )
            else:
                self.status.setText(tr("Ready to review · Save when the portrait looks right."))
            self._refresh_enabled()

        def failed(error):
            self.render_running = False
            if doc is self.doc and doc.revision == revision:
                self.status.setText(str(error))
                self._refresh_enabled()
            else:
                self._schedule_render()

        self._submit(lambda progress, cancel: render_document(snapshot), done, failed, "render")

    def _view_changed(self):
        if self.view_mode.currentIndex() != 2:
            self._tool_changed("pan")
        self._show_preview()

    def _show_preview(self):
        doc = self.doc
        if doc is None:
            return
        mode = self.view_mode.currentIndex()
        markers = []
        if mode == 2 or doc.result is None:
            if self.show_mask.isChecked():
                rgb = np.repeat(doc.alpha[:, :, None], 3, axis=2)
            else:
                # Tint excluded areas so restoring AI mistakes remains possible.
                a = doc.alpha[:, :, None].astype(np.float32) / 255
                rgb = (
                    np.rint(doc.rgb * (0.30 + 0.70 * a) + np.array([65, 12, 22]) * (1 - a))
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            markers = [
                ("N", doc.nose, "#64eab1"),
                ("1", doc.eye_a, "#6cd6ff"),
                ("2", doc.eye_b, "#6cd6ff"),
            ]
            self.canvas.show_image(rgb, markers)
            if self.canvas.tool == "portrait":
                self.canvas.set_tool("pan")
            self.measurements.setText(
                tr(
                    "Source {width} × {height} · pink tint marks removed background",
                    width=doc.rgb.shape[1],
                    height=doc.rgb.shape[0],
                )
            )
        else:
            result = doc.result
            rgb = result["rgb"] if mode == 0 else result["before_rgb"]
            self.canvas.show_image(
                rgb,
                guide=self.guide.isChecked(),
                frame_margins=(
                    doc.settings.headroom,
                    doc.settings.side_margin,
                    doc.settings.bottom_margin,
                ),
                alignment_offset=doc.settings.horizontal_offset + doc.settings.position_x,
            )
            self.canvas.set_tool("portrait")
            self.measurements.setText(
                tr(
                    "948 × 1048 · drag to position · X {x:+d}, Y {y:+d} · {outline}",
                    x=doc.settings.position_x,
                    y=doc.settings.position_y,
                    outline=(
                        tr("strict trim {fraction:.1%}", fraction=result["removed_fraction"])
                        if doc.settings.strict_symmetry
                        else tr("natural outline")
                    ),
                )
            )

    def _tool_changed(self, tool):
        self.tools[tool].setChecked(True)
        self.canvas.set_tool(
            "portrait" if tool == "pan" and self.view_mode.currentIndex() != 2 else tool
        )
        if tool != "pan":
            self.retouch_toggle.setChecked(True)
            self.view_mode.setCurrentIndex(2)
        messages = {
            "pan": tr("Drag to position in the final canvas. Hold Space to pan. Scroll to zoom."),
            "nose": tr("Click the nose tip on the source image."),
            "eye_a": tr("Click the center of one eye on the source image."),
            "eye_b": tr("Click the center of the other eye on the source image."),
            "erase": tr("Paint over unwanted background. Undo restores the last stroke."),
            "restore": tr("Paint to restore original pixels removed by AI or cleanup."),
        }
        self.tool_hint.setText(messages[tool])

    def _begin_position(self):
        doc = self.doc
        if doc is None or not doc.ready or doc.result is None or self.analysis_busy:
            return
        self.timer.stop()
        ys, xs = np.nonzero(doc.result["alpha"])
        if len(xs) == 0:
            return
        nose = doc.nose or ((doc.rgb.shape[1] - 1) / 2, (doc.rgb.shape[0] - 1) / 2)
        t = doc.result["transform"]
        axis = t[0] * nose[0] + t[1] * nose[1] + t[2]
        self.drag_state = (
            doc.settings.position_x,
            doc.settings.position_y,
            (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
            axis,
        )
        doc.changed()
        self._refresh_enabled()

    def _move_position(self, dx, dy, bypass=False):
        if self.drag_state is None or self.doc is None:
            return
        x, y, (left, top, right, bottom), axis = self.drag_state
        if self.snap.isChecked() and not bypass:
            threshold = 10 / max(self.canvas.transform().m11(), 0.01)
            s = self.doc.settings

            def snapped(value, options):
                nearest = min(options, key=lambda target: abs(target - value))
                return nearest if abs(nearest - value) <= threshold else value

            dx = snapped(dx, [473.5 - axis, s.side_margin - left, 948 - s.side_margin - right])
            dy = snapped(dy, [s.headroom - top, 1048 - s.bottom_margin - bottom])
        sx = int(np.clip(round(x + dx), -1896, 1896))
        sy = int(np.clip(round(y + dy), -2096, 2096))
        self.doc.settings.position_x, self.doc.settings.position_y = sx, sy
        self.syncing = True
        self.horizontal_position.setValue(sx)
        self.vertical_position.setValue(sy)
        self.syncing = False
        self.canvas.preview_translation(sx - x, sy - y)
        self.measurements.setText(
            tr("948 × 1048 · X {x:+d}, Y {y:+d} · Alt bypasses snap", x=sx, y=sy)
        )

    def _end_position(self):
        if self.drag_state is not None and self.doc is not None:
            self.drag_state = None
            self.doc.changed()
            self._schedule_render()

    def _reset_position(self):
        if self.doc is not None and not self.analysis_busy:
            self.doc.settings.position_x = self.doc.settings.position_y = 0
            self.doc.changed()
            self._activate(self.doc)

    def _set_point(self, x, y):
        if (
            self.doc is None
            or self.analysis_busy
            or self.canvas.tool not in ("nose", "eye_a", "eye_b")
        ):
            return
        h, w = self.doc.alpha.shape
        point = (min(x, w - 1.0), min(y, h - 1.0))
        setattr(self.doc, self.canvas.tool, point)
        self.doc.changed()
        self._landmark_label()
        self._show_preview()
        self._schedule_render()

    def _begin_stroke(self):
        if self.doc is not None and not self.analysis_busy:
            self.timer.stop()
            self.stroke_before = self.doc.alpha.copy()
            self.previous_brush = None
            self.save_button.setEnabled(False)
            self.next_button.setEnabled(False)

    def _paint(self, x, y):
        if self.doc is None or self.stroke_before is None:
            return
        previous = self.previous_brush or (x, y)
        distance = math.hypot(x - previous[0], y - previous[1])
        steps = max(1, math.ceil(distance / max(1, self.brush_size.value() / 5)))
        for t in np.linspace(1 / steps, 1, steps):
            point = (previous[0] + (x - previous[0]) * t, previous[1] + (y - previous[1]) * t)
            self.doc.paint(point, self.brush_size.value(), self.canvas.tool == "restore")
        self.previous_brush = (x, y)
        self._show_preview()

    def _end_stroke(self):
        if self.doc is not None and self.stroke_before is not None:
            self.doc.record_mask(self.stroke_before)
            self.stroke_before = None
            self.previous_brush = None
            self._schedule_render()

    def _cleanup(self):
        if self.doc is None or self.analysis_busy:
            return
        doc = self.doc
        before = doc.alpha.copy()
        area = self.speck_size.value()
        self._busy(True)
        self.status.setText(tr("Removing small disconnected particles…"))

        def done(alpha):
            doc.alpha = np.asarray(alpha)
            doc.record_mask(before)
            self._show_preview()
            self._schedule_render()

        self.analysis_job = self._submit(
            lambda progress, cancel: _native.clean_specks(before, area, doc.nose),
            done,
            self._failed,
            "cleanup",
        )

    def _inspect_mask(self):
        if self.show_mask.isChecked():
            self.view_mode.setCurrentIndex(2)
        self._show_preview()

    def _undo(self):
        if self.doc is not None and not self.analysis_busy and self.stroke_before is None:
            self.doc.undo()
            self._show_preview()
            self._schedule_render()

    def _redo(self):
        if self.doc is not None and not self.analysis_busy and self.stroke_before is None:
            self.doc.redo()
            self._show_preview()
            self._schedule_render()

    def _reset(self):
        if self.doc is None or self.analysis_busy:
            return
        if (
            QMessageBox.question(
                self,
                tr("Reset photo"),
                tr("Reset the mask, markers, and spacing to their initial values?"),
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.doc.reset()
        self._activate(self.doc)

    def choose_output(self):
        name = QFileDialog.getExistingDirectory(
            self, tr("Choose output folder"), str(self.output or Path.home())
        )
        if name:
            self.output = Path(name)
            self._update_output()

    def _update_output(self):
        self.output_label.setText(
            tr("Saving to  {folder}", folder=str(self.output).replace(str(Path.home()), "~", 1))
        )
        self.output_label.setToolTip(str(self.output))

    def _save(self, advance=False):
        doc = self.doc
        if self.stroke_before is not None or self.analysis_busy or doc is None or not doc.ready:
            return
        if (
            doc.result is None
            or doc.result_revision != doc.revision
            or not np.any(doc.result["alpha"])
        ):
            self.status.setText(tr("Wait for the current preview before saving."))
            return
        try:
            destination = save_image(
                doc.result["rgb"],
                self.output or doc.path.parent / "processed_photos",
                self.export_name.text(),
                self.export_format.currentData(),
                self.export_quality.value(),
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, tr("Could not save"), str(error))
            return
        doc.saved_path = destination
        doc.dirty = False
        index = self.queue.currentRow()
        self.queue.item(index).setText("✓ " + doc.path.name)
        self.status.setText(tr("Saved {name}", name=destination.name))
        if advance and index + 1 < len(self.paths):
            self.queue.setCurrentRow(index + 1)
        elif advance:
            self.status.setText(
                tr("Saved {name} · Last photo in the queue.", name=destination.name)
            )

    def _cancel(self):
        if self.measuring():
            # Stopping a measurement must not pause background preparation.
            self.tasks[self.measure_job][0].cancel.set()
            self.status.setText(tr("Stopping the measurement…"))
            return
        self.prefetch_paused = True
        for identifier in (self.analysis_job, self.batch_job):
            if identifier in self.tasks:
                self.tasks[identifier][0].cancel.set()
        self.status.setText(tr("Stopping preparation…"))

    def closeEvent(self, event):
        if self.tasks:
            answer = QMessageBox.question(
                self,
                tr("Processing in progress"),
                tr("Cancel processing and close when the current operation finishes?"),
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.pending_close = True
                self.prefetch_paused = True
                for task, *_ in self.tasks.values():
                    task.cancel.set()
            event.ignore()
            return
        if not self._confirm_discard():
            event.ignore()
            return
        self.timer.stop()
        self.engine.close()
        self.session_directory.cleanup()
        if self.scan_directory is not None:
            self.scan_directory.cleanup()
        event.accept()


def app_icon():
    image = QPixmap(64, 64)
    image.fill(QColor("#a8e7c7"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#244335"))
    painter.drawEllipse(22, 10, 20, 26)
    painter.drawRoundedRect(12, 37, 40, 22, 12, 12)
    painter.setPen(QColor("#f4fff8"))
    painter.drawLine(32, 5, 32, 59)
    painter.end()
    return QIcon(image)


def main():
    parser = argparse.ArgumentParser(description="Review and standardize human portraits locally.")
    parser.add_argument("photos", nargs="*", type=Path, help="Photo files or folders")
    parser.add_argument("--output", type=Path, help="Destination folder for reviewed photos")
    parser.add_argument("--scan", type=Path, help="Rotate and split a scanned portrait sheet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Fix the interface language before any widget exists; explicit names work without an app.
    settings = QSettings("PortraitProcessor", "PortraitProcessor")
    i18n.set_language(i18n.resolve(settings.value("ui/language", "auto")))
    # OpenCV (a transitive AI dependency) must not override Qt's plugin directory.
    app = QApplication(sys.argv[:1])
    i18n.install_qt_translations(app)
    app.setApplicationName("Portrait Processor")
    app.setDesktopFileName("portrait-processor")
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(STYLE)
    app.setWindowIcon(app_icon())
    window = Window(settings)
    window.show()
    if args.scan:
        window.open_scan_path(args.scan, output=args.output)
    elif args.photos:
        paths = []
        for source in args.photos:
            if source.is_dir():
                paths.extend(
                    sorted(
                        (
                            p
                            for p in source.iterdir()
                            if p.is_file() and p.suffix.lower() in SUPPORTED
                        ),
                        key=lambda p: p.name.casefold(),
                    )
                )
            else:
                paths.append(source)
        if paths:
            window.open_paths(paths)
        else:
            QMessageBox.information(
                window, tr("No photos"), tr("The selected folders contain no supported photos.")
            )
    if args.output:
        window.output = args.output.resolve()
        window._update_output()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
