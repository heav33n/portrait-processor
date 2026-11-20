"""Portable Windows runtime check; no downloaded models or personal photos required."""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from portrait_processor import _native
from portrait_processor.app import STYLE, Window, app_icon, dark_palette
from portrait_processor.document import Document, Face, export_stem, render_document, save_image
from portrait_processor.scan_dialog import ScanReview
from portrait_processor.scans import load_scan
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

output = Path(
    os.environ.get("PORTRAIT_SELFTEST_OUTPUT", str(Path(tempfile.gettempdir()) / "portrait-test"))
)
output.mkdir(parents=True, exist_ok=True)
rgb = np.full((600, 800, 3), (230, 235, 240), dtype=np.uint8)
rgb[40:560, 160:640] = (185, 115, 95)
source = output / "Test source.png"
Image.fromarray(rgb).save(source)
doc = Document.load(source)
alpha = np.zeros((600, 800), dtype=np.uint8)
alpha[40:560, 160:640] = 255
doc.apply_analysis(alpha, [Face((399.5, 220), (330, 160), (470, 160))])
doc.settings.horizontal_offset = 45
result = render_document(doc)
assert result["rgb"].shape == (1048, 948, 3)
assert np.array_equal(result["alpha"], result["before_alpha"])
t = result["transform"]
assert abs(t[0] * 399.5 + t[1] * 220 + t[2] - 518.5) < 1e-6
assert np.any(result["alpha"][-1] >= 250)
saved = []
for image_format in ["PNG", "JPEG", "WEBP", "TIFF", "BMP"]:
    path = save_image(result["rgb"], output, f"My portrait {image_format}", image_format, 94)
    with Image.open(path) as image:
        assert image.size == (948, 1048)
        assert image.format == image_format
    saved.append(path.name)
first = save_image(result["rgb"], output, "Collision", "PNG")
second = save_image(result["rgb"], output, "Collision", "PNG")
assert first != second
noise = np.random.default_rng(17).integers(0, 256, (1048, 948, 3), dtype=np.uint8)
low = save_image(noise, output, "Texture low quality", "JPEG", 15)
high = save_image(noise, output, "Texture high quality", "JPEG", 95)
assert high.stat().st_size > low.stat().st_size
try:
    export_stem("CON.png")
except ValueError:
    pass
else:
    raise AssertionError("Reserved Windows filename accepted")
app = QApplication([])
app.setStyle("Fusion")
app.setPalette(dark_palette())
app.setStyleSheet(STYLE)
app.setWindowIcon(app_icon())
window = Window(QSettings(str(output / "settings.ini"), QSettings.Format.IniFormat))
doc.result = result
doc.result_revision = doc.revision
window.paths = [doc.path]
window.documents[doc.path] = doc
window.output = output
window.queue.addItem(doc.path.name)
window.queue.setCurrentRow(0)
window.queue_heading.setText("YOUR PHOTOS  ·  1")
window.export_format.setCurrentIndex(window.export_format.findData("JPEG"))
window.export_name.setText("Windows GUI export.jpg")
window.show()
app.processEvents()
assert window.manual_button.isEnabled()
window.manual_button.click()
window.horizontal_position.setValue(65)
deadline = time.monotonic() + 20
while window.doc.result_revision != window.doc.revision:
    app.processEvents()
    if time.monotonic() > deadline:
        raise AssertionError("Manual positioning preview did not finish")
    time.sleep(0.01)
window.view_mode.setCurrentIndex(0)
transform = window.doc.result["transform"]
assert abs(transform[0] * 399.5 + transform[1] * 220 + transform[2] - 583.5) < 1e-6
window.save_button.click()
assert window.doc.saved_path.suffix == ".jpg"
window.grab().save(str(output / "Windows-preview.png"))
doc.dirty = False
window.close()

# Exercise Windows OpenCV detection, crop review, and the complete queue workflow.
sheet = Image.new("RGB", (1000, 1200), (237, 238, 237))
draw = ImageDraw.Draw(sheet)
for row in range(2):
    for col in range(2):
        x, y = 30 + col * 490, 25 + row * 590
        draw.rectangle(
            (x, y, x + 425, y + 525), fill=(245, 246, 245), outline=(170, 170, 170), width=2
        )
        draw.ellipse((x + 110, y + 70, x + 320, y + 350), fill=(175, 120, 95))
        draw.rectangle((x + 30, y + 350, x + 395, y + 520), fill=(40, 80, 60))
scan_path = output / "Windows scanned sheet.png"
sheet.transpose(Image.Transpose.ROTATE_180).save(scan_path)
scan = load_scan(scan_path)
assert len(scan.crops) == 4
assert np.array_equal(scan.rgb, np.asarray(sheet))
review = ScanReview(scan)
review.show()
for _ in range(5):
    app.processEvents()
    time.sleep(0.02)
review.grab().save(str(output / "Windows-scan-review.png"))
review.close()


class TestEngine:
    calls = 0

    def analyze(self, rgb, original_alpha, progress, cancel):
        self.calls += 1
        h, w = rgb.shape[:2]
        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[h // 8 :, w // 5 : w * 4 // 5] = 255
        return alpha, [Face((w / 2, h * 0.4), (w * 0.4, h * 0.3), (w * 0.6, h * 0.3))], []

    def close(self):
        pass


batch = Window(QSettings(str(output / "batch-settings.ini"), QSettings.Format.IniFormat))
batch.engine = TestEngine()
batch.show()
batch._import_scan(scan, scan.crops, output=output / "scan-exports")
deadline = time.monotonic() + 40
while batch.tasks or batch.doc is None or batch.doc.result_revision != batch.doc.revision:
    app.processEvents()
    if time.monotonic() > deadline:
        raise AssertionError(f"Scan batch did not finish: {batch.status.text()}")
    time.sleep(0.01)
assert batch.engine.calls == 4
assert len(batch.paths) == 4
assert batch.save_button.isEnabled()
batch._save(True)
deadline = time.monotonic() + 20
while (
    batch.tasks
    or batch.doc is None
    or batch.doc.path != batch.paths[1]
    or batch.doc.result_revision != batch.doc.revision
):
    app.processEvents()
    if time.monotonic() > deadline:
        raise AssertionError("Second scanned portrait did not load")
    time.sleep(0.01)
assert batch.engine.calls == 4
for item in batch.documents.values():
    item.dirty = False
batch.draft_dirty = {p: False for p in batch.draft_dirty}
batch.close()

from portrait_processor.scanner_dialog import ScannerDialog


class TestScanner:
    def devices(self, driver, cancel):
        return ["Canon CanoScan LiDE 300"]

    def capture(self, settings, path, cancel, *, preview=False):
        Image.new("RGB", (200, 300), (140, 95, 75)).save(path)
        return path


scanner_preferences = QSettings(str(output / "scanner-settings.ini"), QSettings.Format.IniFormat)
scanner_preferences.setValue("scanner/folder", str(output / "direct-scans"))
panel = ScannerDialog(preferences=scanner_preferences, backend=TestScanner())
panel.show()
deadline = time.monotonic() + 20
while panel.job is not None or panel.device.count() != 1:
    app.processEvents()
    if time.monotonic() > deadline:
        raise AssertionError("Scanner discovery UI did not finish")
    time.sleep(0.01)
assert panel.scale.value() == 100
assert panel.dpi.currentData() == 1200
panel.preview_button.click()
while panel.job is not None:
    app.processEvents()
    time.sleep(0.01)
panel.grab().save(str(output / "Windows-direct-scanner.png"))
panel.scale.setValue(49)
panel.name.setText("My directly scanned sheet")
panel.scan_button.click()
while panel.accepted_path is None:
    app.processEvents()
    if time.monotonic() > deadline:
        raise AssertionError("Scanner save UI did not finish")
    time.sleep(0.01)
with Image.open(panel.accepted_path) as saved_sheet:
    # 1200 DPI at 49% is read at 600 DPI, then resized by 588/600.
    assert saved_sheet.size == (196, 294), saved_sheet.size
import cv2
import onnx
import onnxruntime
from portrait_processor.ai import vision_modules
from portrait_processor.levels import LEVELS
from portrait_processor.models import bundled_directory, expand_model

mediapipe, _, _ = vision_modules()
assert hasattr(cv2, "ximgproc"), "Level 2 needs OpenCV's guided filter (opencv-contrib)"
bundle = bundled_directory()
bundled = sorted({item.model for item in LEVELS if item.bundled})
missing = [m for m in bundled if not (bundle / f"{m}.fp16.onnx").is_file()]
assert not missing, f"Bundled models missing from {bundle}: {missing}"
expanded = output / "u2net_human_seg.onnx"
expand_model(bundle / "u2net_human_seg.fp16.onnx", expanded)
session = onnxruntime.InferenceSession(str(expanded), providers=["CPUExecutionProvider"])
probe = np.zeros((1, 3, 320, 320), np.float32)
session.run(None, {session.get_inputs()[0].name: probe})
del session
expanded.unlink()

report = {
    "status": "passed",
    "native_module": _native.__file__,
    "formats": saved,
    "onnxruntime": onnxruntime.__version__,
    "onnx": onnx.__version__,
    "bundled_models": bundled,
    "mediapipe": mediapipe.__version__,
    "scan_portraits": 4,
    "scanner_panel": "passed with simulated acquisition",
}
(output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report), flush=True)
