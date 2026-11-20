"""Scanner controls and a low-resolution preview in the portrait editor's dark UI."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSettings, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .ai import Cancelled
from .canvas import Canvas
from .i18n import tr
from .preferences import int_setting
from .scanner import ScannerBackend, ScannerSettings, save_capture, scan_preview


class ScanSignals(QObject):
    progress = Signal(str)
    finished = Signal(object)


class ScanJob(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.signals = ScanSignals()
        self.cancel = threading.Event()
        self.function = function

    @Slot()
    def run(self):
        try:
            result = self.function(self.signals.progress.emit, self.cancel)
            payload = result, None
        except Exception as error:  # noqa: BLE001 - send worker errors to the UI
            payload = None, error
        self.signals.finished.emit(payload)


class ScannerDialog(QDialog):
    def __init__(self, parent=None, *, preferences=None, backend=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Scanners · Portrait Processor"))
        self.resize(1120, 820)
        self.setMinimumSize(880, 650)
        self.preferences = (
            preferences
            if preferences is not None
            else QSettings("PortraitProcessor", "PortraitProcessor")
        )
        self.backend = backend
        self.directory = tempfile.TemporaryDirectory(prefix="portrait-scanner-")
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.job = None
        self.pending_close = False
        self.accepted_path = None
        self.output = Path(
            str(
                self.preferences.value(
                    "scanner/folder", str(Path.home() / "Pictures" / "Portrait Scans")
                )
            )
        )
        body = QVBoxLayout(self)
        body.setContentsMargins(24, 22, 24, 22)
        body.setSpacing(16)
        title = QLabel(tr("Scan your portraits"))
        title.setObjectName("title")
        body.addWidget(title)
        description = QLabel(
            tr("CanoScan LiDE 300 · Connect by USB. Canon's scanner driver must be installed.")
        )
        description.setObjectName("muted")
        description.setWordWrap(True)
        body.addWidget(description)
        row = QHBoxLayout()
        row.setSpacing(20)
        self.preview = Canvas()
        self.preview.scene().clear()
        text = self.preview.scene().addText(
            tr("Place the portrait sheet on the glass.\n\nChoose Preview to check its position.")
        )
        text.setDefaultTextColor(QColor("#b5c6bb"))
        self.preview.scene().setSceneRect(text.boundingRect().adjusted(-60, -170, 60, 170))
        self.preview.set_tool("pan")
        row.addWidget(self.preview, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350)
        sidebar = QWidget()
        controls = QVBoxLayout(sidebar)
        controls.setContentsMargins(0, 0, 8, 0)
        controls.setSpacing(14)
        form = QFormLayout()
        form.setVerticalSpacing(12)
        self.device = QComboBox()
        self.device.setMinimumContentsLength(15)
        self.refresh = QPushButton(tr("Find scanners"))
        self.refresh.clicked.connect(self.find_devices)
        form.addRow(tr("Scanner"), self.device)
        form.addRow("", self.refresh)
        self.driver = QComboBox()
        if sys.platform == "win32":
            connections = [
                (tr("TWAIN · full resolution"), "twain"),
                (tr("WIA · up to 600 DPI"), "wia"),
            ]
        elif sys.platform == "darwin":
            connections = [("Apple", "apple")]
        else:
            connections = [("SANE", "sane")]
        for label, data in connections:
            self.driver.addItem(label, data)
        self.driver.setCurrentIndex(
            max(
                0,
                self.driver.findData(
                    self.preferences.value("scanner/driver", self.driver.currentData())
                ),
            )
        )
        form.addRow(tr("Connection"), self.driver)
        self.dpi = QComboBox()
        form.addRow(tr("Resolution"), self.dpi)
        self.scale = QSpinBox()
        self.scale.setRange(1, 100)
        self.scale.setSuffix(" %")
        self.scale.setValue(int_setting(self.preferences, "scanner/scale", 100))
        self.scale.setToolTip(
            tr(
                "100% keeps the scanned pixels. 49% makes each dimension 49% as large. This is not JPEG quality."
            )
        )
        form.addRow(tr("Output scale"), self.scale)
        self.page = QComboBox()
        for label, data in [
            (tr("Full bed · 216 × 297 mm"), "216x297mm"),
            ("A4", "a4"),
            (tr("Letter"), "letter"),
        ]:
            self.page.addItem(label, data)
        form.addRow(tr("Scan area"), self.page)
        self.color = QComboBox()
        for label, data in [(tr("Color · 24-bit"), "color"), (tr("Grayscale"), "gray")]:
            self.color.addItem(label, data)
        form.addRow(tr("Color"), self.color)
        self.estimate = QLabel()
        self.estimate.setObjectName("muted")
        self.estimate.setWordWrap(True)
        form.addRow(self.estimate)
        self.name = QLineEdit(datetime.now().strftime("Sheet_%Y%m%d_%H%M%S"))
        form.addRow(tr("Scan name"), self.name)
        self.image_format = QComboBox()
        for label, data in [
            (tr("PNG · lossless"), "PNG"),
            (tr("TIFF · lossless"), "TIFF"),
            (tr("JPG · quality 95"), "JPEG"),
        ]:
            self.image_format.addItem(label, data)
        form.addRow(tr("Save scan as"), self.image_format)
        controls.addLayout(form)
        self.folder = QLabel()
        self.folder.setWordWrap(True)
        self.folder.setObjectName("muted")
        self._folder_label()
        controls.addWidget(self.folder)
        self.choose_folder = QPushButton(tr("Change scan folder…"))
        self.choose_folder.clicked.connect(self._choose_folder)
        controls.addWidget(self.choose_folder)
        help_text = QLabel(
            tr(
                "Preview uses 150 DPI. Scanning uses your chosen resolution and scale. The sheet is saved first, then opens for portrait separation and batches of five. Portrait names and formats stay editable."
            )
        )
        help_text.setObjectName("muted")
        help_text.setWordWrap(True)
        controls.addWidget(help_text)
        controls.addStretch()
        scroll.setWidget(sidebar)
        row.addWidget(scroll)
        body.addLayout(row, 1)
        self.status = QLabel(tr("Looking for your scanner…"))
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        body.addWidget(self.status)
        footer = QHBoxLayout()
        self.close_button = QPushButton(tr("Close"))
        self.close_button.clicked.connect(self.reject)
        self.preview_button = QPushButton(tr("Preview"))
        self.preview_button.clicked.connect(self._preview)
        self.scan_button = QPushButton(tr("Scan & import portraits"))
        self.scan_button.setObjectName("primary")
        self.scan_button.clicked.connect(self._scan)
        footer.addWidget(self.close_button)
        footer.addStretch()
        footer.addWidget(self.preview_button)
        footer.addWidget(self.scan_button)
        body.addLayout(footer)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._elapsed)
        self.driver.currentIndexChanged.connect(self._connection_changed)
        for control in (self.dpi, self.page):
            control.currentIndexChanged.connect(self._estimate)
        self.scale.valueChanged.connect(self._estimate)
        self.device.currentIndexChanged.connect(self._enabled)
        self._resolution_choices()
        self._enabled()
        QTimer.singleShot(0, self.find_devices)

    def _resolution_choices(self):
        previous = self.dpi.currentData() or int_setting(self.preferences, "scanner/dpi", 1200)
        self.dpi.clear()
        for dpi in (150, 300, 600, 1200):
            if self.driver.currentData() != "wia" or dpi <= 600:
                label = f"{dpi} DPI"
                self.dpi.addItem(label, dpi)
        self.dpi.setCurrentIndex(
            max(
                0,
                self.dpi.findData(
                    min(previous, 600) if self.driver.currentData() == "wia" else previous
                ),
            )
        )
        self._estimate()

    def _connection_changed(self):
        self._resolution_choices()
        self.find_devices()

    def _estimate(self):
        dpi = self.dpi.currentData() or 1200
        width, height = {"216x297mm": (216, 297), "a4": (210, 297), "letter": (215.9, 279.4)}[
            self.page.currentData()
        ]
        width, height = (round(n / 25.4 * dpi * self.scale.value() / 100) for n in (width, height))
        reads = ScannerSettings("", dpi=dpi, scale=self.scale.value()).acquisition_dpi
        self.estimate.setText(
            tr(
                "Approximately {width:,} × {height:,} pixels. The scanner reads at {reads} DPI; "
                "higher reading resolution takes longer.",
                width=width,
                height=height,
                reads=reads,
            )
        )

    def _folder_label(self):
        self.folder.setText(tr("Scan folder: {folder}", folder=str(self.output)))

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, tr("Save scanned sheets"), str(self.output))
        if path:
            self.output = Path(path)
            self._folder_label()

    def _enabled(self, *_):
        busy = self.job is not None
        for control in (
            self.device,
            self.driver,
            self.dpi,
            self.scale,
            self.page,
            self.color,
            self.name,
            self.image_format,
            self.choose_folder,
            self.refresh,
        ):
            control.setEnabled(not busy)
        self.scan_button.setEnabled(not busy and bool(self.device.currentText()))
        self.preview_button.setEnabled(not busy and bool(self.device.currentText()))
        self.close_button.setText(tr("Cancel") if busy else tr("Close"))

    def _status(self, text):
        self.message = text
        self.status.setText(text)

    def _elapsed(self):
        seconds = int(time.monotonic() - self.started)
        self.status.setText(
            tr(
                "{message} · {minutes}:{seconds:02d} elapsed",
                message=self.message,
                minutes=seconds // 60,
                seconds=seconds % 60,
            )
        )

    def _start(self, function, completed):
        if self.job is not None:
            return
        self.job = ScanJob(function)
        self.job.signals.progress.connect(self._status)
        self.completed_callback = completed
        self.job.signals.finished.connect(self._completed)
        self.started = time.monotonic()
        self.timer.start()
        self._enabled()
        self.pool.start(self.job)

    @Slot(object)
    def _completed(self, payload):
        self._finished(payload, self.completed_callback)

    def _finished(self, payload, completed):
        self.job = None
        self.timer.stop()
        self._enabled()
        result, error = payload
        if self.pending_close:
            self.reject()
        elif error is not None:
            self._status(tr("Scanning cancelled.") if isinstance(error, Cancelled) else str(error))
        else:
            completed(result)

    def find_devices(self):
        if self.job is not None:
            return
        try:
            self.backend = self.backend or ScannerBackend()
        except RuntimeError as error:
            self._status(str(error))
            return
        backend = self.backend
        driver = self.driver.currentData()
        self.device.clear()
        self._status(tr("Finding connected scanners…"))

        def found(devices):
            self.device.addItems(devices)
            preferred = next((i for i, name in enumerate(devices) if "lide 300" in name.lower()), 0)
            self.device.setCurrentIndex(preferred)
            self._status(
                tr("Scanner ready. Preview the sheet or start a scan.")
                if devices
                else tr(
                    "No scanner found. Connect the LiDE 300 by USB, install Canon's driver, then choose Find scanners. You can also try the other Connection option."
                )
            )
            self._enabled()

        self._start(lambda progress, cancel: backend.devices(driver, cancel), found)

    def settings(self):
        return ScannerSettings(
            self.device.currentText(),
            self.driver.currentData(),
            self.dpi.currentData(),
            self.scale.value(),
            self.page.currentData(),
            self.color.currentData(),
        )

    def _preview(self):
        settings = self.settings()
        self._status(tr("Scanning a quick preview…"))
        backend = self.backend
        assert backend is not None
        path = Path(self.directory.name) / "preview.tiff"
        path.unlink(missing_ok=True)

        def acquire(progress, cancel):
            backend.capture(settings, path, cancel, preview=True)
            return scan_preview(path)

        def shown(rgb):
            self.preview.show_image(rgb)
            self._status(
                tr("Preview ready. The final scan will use your chosen resolution and scale.")
            )

        self._start(acquire, shown)

    def _scan(self):
        settings = self.settings()
        try:
            settings.validate()
            from .document import export_stem

            export_stem(self.name.text())
        except ValueError as error:
            self._status(str(error))
            return
        name, image_format, directory = (
            self.name.text(),
            self.image_format.currentData(),
            self.output,
        )
        self._status(tr("Scanning the full sheet…"))
        backend = self.backend
        assert backend is not None
        path = Path(self.directory.name) / "capture.tiff"
        path.unlink(missing_ok=True)

        def acquire(progress, cancel):
            backend.capture(settings, path, cancel)
            progress(tr("Saving the scanned sheet…"))
            return save_capture(path, directory, name, settings, image_format, cancel)

        def saved(path):
            for key, value in {
                "folder": str(directory),
                "dpi": settings.dpi,
                "scale": settings.scale,
                "driver": settings.driver,
            }.items():
                self.preferences.setValue("scanner/" + key, value)
            self.accepted_path = path
            self.accept()

        self._start(acquire, saved)

    def done(self, result):
        if self.job is not None:
            self.pending_close = True
            self.job.cancel.set()
            self._status(tr("Stopping the scanner…"))
            return
        self.timer.stop()
        self.directory.cleanup()
        super().done(result)

    def closeEvent(self, event):
        if self.job is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
