"""Exercise the actual desktop worker, capture its result, and save a sample PNG."""

import sys
import tempfile
from pathlib import Path

from PIL import Image
from portrait_processor.app import STYLE, Window, app_icon
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication
from skimage import data

artifacts = Path(__file__).resolve().parents[1] / "artifacts"
artifacts.mkdir(exist_ok=True)
source = artifacts / "astronaut_source.png"
Image.fromarray(data.astronaut()).save(source)
preferences_directory = tempfile.TemporaryDirectory(prefix="portrait-qa-")
app = QApplication([])
app.setStyle("Fusion")
app.setStyleSheet(STYLE)
app.setWindowIcon(app_icon())
window = Window(
    QSettings(str(Path(preferences_directory.name) / "settings.ini"), QSettings.Format.IniFormat)
)
window.show()
window.open_paths([source])
window.output = artifacts / "desktop-smoke" / "processed_photos"
window._update_output()
success = False
timer = QTimer()


def inspect():
    global success
    if window.doc is None or not window.save_button.isEnabled():
        return
    timer.stop()
    assert len(window.doc.faces) == 1, window.status.text()
    for mode, name in [(0, "editor_result.png"), (2, "editor_source.png")]:
        window.view_mode.setCurrentIndex(mode)
        app.processEvents()
        assert window.grab().save(str(artifacts / name))
    window.view_mode.setCurrentIndex(0)
    window.export_name.setText("Desktop test portrait.png")
    window.save_button.click()
    assert window.doc.saved_path is not None
    assert window.doc.saved_path.is_file()
    assert window.doc.saved_path.name.startswith("Desktop test portrait")
    assert not window.doc.saved_path.name.endswith(".png.png")
    assert not window.doc.dirty
    print(f"Desktop AI, preview, and Save passed: {window.doc.saved_path}", flush=True)
    success = True
    window.close()
    app.quit()


timer.timeout.connect(inspect)
timer.start(250)
QTimer.singleShot(150000, app.quit)
app.exec()
preferences_directory.cleanup()
sys.exit(0 if success else 1)
