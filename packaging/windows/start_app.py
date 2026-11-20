"""Start the private Windows runtime without depending on an installed Python."""

import ctypes
import faulthandler
import logging
import os
import runpy
import sys
from pathlib import Path

runtime = Path(__file__).resolve().parent
packages = runtime / "Lib" / "site-packages"
local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
state = local / "PortraitProcessor"
logs = state / "logs"
logs.mkdir(parents=True, exist_ok=True)
stream = (logs / "app.log").open("a", encoding="utf-8", buffering=1)
if sys.stdout is None:
    sys.stdout = stream
if sys.stderr is None:
    sys.stderr = stream
logging.basicConfig(
    level=logging.INFO, stream=stream, format="%(asctime)s %(levelname)s %(message)s"
)
faulthandler.enable(file=stream)
os.environ.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 2)))
os.environ.setdefault("NUMBA_CACHE_DIR", str(state / "Cache" / "numba"))
os.environ["QT_PLUGIN_PATH"] = str(packages / "PySide6" / "plugins")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(packages / "PySide6" / "plugins" / "platforms")
_dll_handles = []
for folder in [
    runtime,
    packages / "PySide6",
    packages / "shiboken6",
    packages / "onnxruntime" / "capi",
    packages / "mediapipe" / "tasks" / "c",
]:
    if folder.is_dir():
        _dll_handles.append(os.add_dll_directory(str(folder)))
try:
    if "--self-test" in sys.argv:
        runpy.run_path(str(runtime / "self_test.py"), run_name="__main__")
    else:
        from portrait_processor.app import main

        main()
except SystemExit:
    raise
except BaseException:
    logging.getLogger(__name__).exception("Application failed")
    ctypes.windll.user32.MessageBoxW(
        None,
        f"Portrait Processor could not start. Details were saved to:\n{logs / 'app.log'}",
        "Portrait Processor",
        0x10,
    )
    raise SystemExit(1)
