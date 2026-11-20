"""Exercise real sheet detection, crop review, batch AI, and exports in the desktop app."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw
from portrait_processor.app import STYLE, Window, dark_palette
from portrait_processor.scan_dialog import ScanReview
from portrait_processor.scans import load_scan
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

parser = argparse.ArgumentParser()
parser.add_argument("scan", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
source_hash = hashlib.sha256(args.scan.read_bytes()).hexdigest()
scan = load_scan(args.scan, print)
print(f"Detected {len(scan.crops)} portraits", flush=True)
app = QApplication([])
app.setStyle("Fusion")
app.setPalette(dark_palette())
app.setStyleSheet(STYLE)
window = Window(QSettings(str(output / "test-settings.ini"), QSettings.Format.IniFormat))
window.show()
review = ScanReview(scan, window)
review.show()
for _ in range(10):
    app.processEvents()
    time.sleep(0.02)
review.grab().save(str(output / "crop-review.png"))
review.close()
window._import_scan(scan, scan.crops, output=output / "processed_photos")
start = time.monotonic()
previous = ""
while window.tasks or window.doc is None or window.doc.result_revision != window.doc.revision:
    app.processEvents()
    status = re.sub(r" · \d+:\d{2} elapsed$", "", window.status.text())
    if status != previous:
        print(status, flush=True)
        previous = status
    if not window.tasks and not window.analysis_busy and window.doc is None:
        raise RuntimeError(status)
    if time.monotonic() - start > 2400:
        raise TimeoutError("Scan processing did not finish")
    time.sleep(0.02)
reports = []
first_batch_seconds = round(time.monotonic() - start, 1)
window._process_all()
while window.tasks or window.doc is None or window.doc.result_revision != window.doc.revision:
    app.processEvents()
    status = re.sub(r" · \d+:\d{2} elapsed$", "", window.status.text())
    if status != previous:
        print(status, flush=True)
        previous = status
    if time.monotonic() - start > 2400:
        raise TimeoutError("Explicit scan processing did not finish")
    time.sleep(0.02)
contact = Image.new("RGB", (1000, ((len(window.paths) + 3) // 4) * 300), "#151c1b")
draw = ImageDraw.Draw(contact)
for index, path in enumerate(window.paths):
    window.queue.setCurrentRow(index)
    while (
        window.tasks
        or window.doc is None
        or window.doc.path != path
        or (window.doc.ready and window.doc.result_revision != window.doc.revision)
    ):
        app.processEvents()
        time.sleep(0.02)
    doc = window.doc
    report = {"name": path.name, "faces": len(doc.faces), "ready": bool(doc.ready)}
    if doc.ready:
        window.export_format.setCurrentIndex(window.export_format.findData("PNG"))
        window._save()
        assert doc.saved_path is not None, window.status.text()
        report["saved"] = str(doc.saved_path)
        im = Image.fromarray(doc.result["rgb"])
        im.thumbnail((235, 270))
        x = index % 4 * 250 + (250 - im.width) // 2
        y = index // 4 * 300
        contact.paste(im, (x, y))
        draw.text((index % 4 * 250 + 12, y + 277), str(index + 1), fill="white")
        if index == 0:
            window.grab().save(str(output / "batch-review.png"))
    reports.append(report)
    print(json.dumps(report), flush=True)
contact.save(output / "processed-contact-sheet.jpg", quality=95)
assert hashlib.sha256(args.scan.read_bytes()).hexdigest() == source_hash
report = {
    "portraits": reports,
    "elapsed_seconds": round(time.monotonic() - start, 1),
    "original_unchanged": True,
    "first_batch_seconds": first_batch_seconds,
    "level": window.engine.level,
    "model": window.engine.model_name,
}
(output / "report.json").write_text(json.dumps(report, indent=2))
for doc in window.documents.values():
    doc.dirty = False
window.draft_dirty = {p: False for p in window.draft_dirty}
window.close()
print("Scan verification complete", flush=True)
