import json
import os
import sys
import threading
import xml.etree.ElementTree as ET

import pytest
from PIL import Image
from portrait_processor.ai import Cancelled, check_cancel
from portrait_processor.scanner import ScannerBackend, ScannerSettings, save_capture
from portrait_processor.scanner_dialog import ScannerDialog
from PySide6.QtCore import QSettings


@pytest.fixture
def scanner_process(tmp_path):
    script = tmp_path / "scanner.py"
    log = tmp_path / "arguments.json"
    script.write_text("""
import json,os,sys,time
from pathlib import Path
from PIL import Image
args=sys.argv[2:]
Path(sys.argv[1]).write_text(json.dumps(args))
data=os.environ.get('NAPS2_TEST_DATA')
Path(sys.argv[1]).with_suffix('.env').write_text(data or '')
if '--listdevices' in args:
    print('Canon CanoScan LiDE 300')
elif 'slow' in args:
    time.sleep(30)
else:
    Image.new('RGB',(200,300),(130,90,60)).save(args[args.index('--output')+1])
""")
    return ScannerBackend(
        [sys.executable, str(script), str(log)], data_directory=tmp_path / "naps2"
    ), log


def test_scanner_uses_explicit_native_resolution_and_no_driver_gui(scanner_process, tmp_path):
    backend, log = scanner_process
    cancel = threading.Event()
    assert backend.devices("twain", cancel) == ["Canon CanoScan LiDE 300"]
    settings = ScannerSettings("Canon CanoScan LiDE 300")
    result = backend.capture(settings, tmp_path / "capture.tiff", cancel)
    assert result.is_file()
    args = json.loads(log.read_text())
    assert args[args.index("--dpi") + 1] == "1200"
    assert args[args.index("--device") + 1] == settings.device
    assert "--noprofile" not in args
    assert args[args.index("--profile") + 1] == "PortraitProcessor"
    assert "--progress" not in args
    backend.capture(settings, tmp_path / "preview.tiff", cancel, preview=True)
    args = json.loads(log.read_text())
    assert args[args.index("--dpi") + 1] == "150"


@pytest.mark.parametrize(
    "dpi,scale,reads",
    [
        (1200, 49, 600),
        (1200, 50, 600),
        (1200, 51, 1200),
        (600, 49, 300),
        (300, 100, 300),
        (150, 10, 150),
    ],
)
def test_scaled_scans_read_the_glass_at_the_lowest_sufficient_resolution(
    scanner_process, tmp_path, dpi, scale, reads
):
    backend, log = scanner_process
    settings = ScannerSettings("Canon CanoScan LiDE 300", dpi=dpi, scale=scale)
    assert settings.acquisition_dpi == reads
    backend.capture(settings, tmp_path / "capture.tiff", threading.Event())
    args = json.loads(log.read_text())
    assert args[args.index("--dpi") + 1] == str(reads)


def test_scanner_uses_a_private_lossless_naps2_profile(scanner_process, tmp_path):
    backend, log = scanner_process
    backend.capture(ScannerSettings("Canon"), tmp_path / "capture.tiff", threading.Event())
    data = tmp_path / "naps2"
    assert log.with_suffix(".env").read_text() == str(data)
    profiles = ET.parse(data / "profiles.xml").getroot()
    assert profiles.tag == "ArrayOfScanProfile"
    (profile,) = profiles.findall("ScanProfile")
    assert profile.findtext("DisplayName") == "PortraitProcessor"
    assert profile.findtext("MaxQuality") == "true"
    assert profile.findtext("Version") == "2"
    assert "NAPS2_TEST_DATA" not in os.environ


def test_scanner_cancellation_terminates_the_acquisition_process(scanner_process, tmp_path):
    backend, _ = scanner_process
    cancel = threading.Event()
    timer = threading.Timer(0.5, cancel.set)
    timer.start()
    try:
        with pytest.raises(Cancelled):
            backend.capture(ScannerSettings("slow"), tmp_path / "cancelled.tiff", cancel)
    finally:
        timer.cancel()
    assert not (tmp_path / "cancelled.tiff").exists()


@pytest.mark.parametrize(
    "image_format,extension", [("PNG", "png"), ("TIFF", "tiff"), ("JPEG", "jpg")]
)
def test_capture_scale_format_and_collision_do_not_modify_source(tmp_path, image_format, extension):
    source = tmp_path / "raw.tiff"
    Image.new("RGB", (200, 300), (130, 90, 60)).save(source)
    original = source.read_bytes()
    # 1200 DPI at 49% is read at 600 DPI, then resized by 588/600.
    settings = ScannerSettings("Canon", scale=49)
    path = save_capture(source, tmp_path, "My sheet", settings, image_format, threading.Event())
    second = save_capture(source, tmp_path, "My sheet", settings, image_format, threading.Event())
    assert path.name == f"My sheet.{extension}"
    assert second.name == f"My sheet_2.{extension}"
    with Image.open(path) as image:
        assert image.size == (196, 294)
        assert image.format == image_format
        if image_format != "JPEG":
            assert round(image.info["dpi"][0]) == 588
    assert source.read_bytes() == original


def test_wia_cannot_silently_accept_unsupported_1200_dpi():
    with pytest.raises(ValueError, match="600 DPI"):
        ScannerSettings("Canon", driver="wia", dpi=1200).validate()


class FakeScanner:
    def __init__(self):
        self.calls = []

    def devices(self, driver, cancel):
        return ["Canon CanoScan LiDE 300"]

    def capture(self, settings, path, cancel, *, preview=False):
        check_cancel(cancel)
        self.calls.append((settings, preview))
        Image.new("RGB", (200, 300), (130, 90, 60)).save(path)
        return path


def test_scanner_panel_previews_then_saves_and_returns_the_sheet(qtbot, tmp_path):
    backend = FakeScanner()
    preferences = QSettings(str(tmp_path / "scanner.ini"), QSettings.Format.IniFormat)
    preferences.setValue("scanner/folder", str(tmp_path / "captures"))
    dialog = ScannerDialog(preferences=preferences, backend=backend)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.job is None and dialog.device.count() == 1)
    assert dialog.scale.value() == 100
    assert dialog.dpi.currentData() == 1200
    dialog.preview_button.click()
    qtbot.waitUntil(lambda: dialog.job is None)
    assert backend.calls[0][1] is True
    assert dialog.accepted_path is None
    dialog.scale.setValue(49)
    dialog.name.setText("Scanned portraits")
    dialog.scan_button.click()
    qtbot.waitUntil(lambda: dialog.accepted_path is not None)
    assert backend.calls[1][1] is False
    assert backend.calls[1][0].acquisition_dpi == 600
    assert dialog.accepted_path is not None
    with Image.open(dialog.accepted_path) as image:
        assert image.size == (196, 294)
    assert dialog.accepted_path.name == "Scanned portraits.png"
    assert not dialog.isVisible()


def test_scanner_panel_closing_during_discovery_cancels_cleanly(qtbot, tmp_path):
    class SlowScanner(FakeScanner):
        def devices(self, driver, cancel):
            cancel.wait(4)
            check_cancel(cancel)
            return []

    dialog = ScannerDialog(
        preferences=QSettings(str(tmp_path / "scanner.ini"), QSettings.Format.IniFormat),
        backend=SlowScanner(),
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.job is not None)
    dialog.reject()
    qtbot.waitUntil(lambda: dialog.job is None and not dialog.isVisible())
