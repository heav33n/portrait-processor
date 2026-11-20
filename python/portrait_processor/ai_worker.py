"""Private persistent model process. Only local filenames cross the pipe."""

import json
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

# DLL search-directory handles belong to a process and are not inherited on Windows.
_dll_handles = []
if sys.platform == "win32":
    runtime = Path(sys.executable).parent
    packages = Path(__file__).resolve().parent.parent
    for directory in (
        runtime,
        packages / "onnxruntime" / "capi",
        packages / "mediapipe" / "tasks" / "c",
    ):
        if directory.is_dir():
            _dll_handles.append(os.add_dll_directory(str(directory)))

import numpy as np

from .ai import InferenceEngine


def main():
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def emit(payload):
        print(json.dumps(payload), file=protocol, flush=True)

    engine = InferenceEngine(int(os.environ.get("PORTRAIT_AI_LEVEL", "1")))
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)

                def progress(text):
                    emit({"event": "progress", "text": text})

                engine.set_level(int(request.get("level", engine.level)))
                with np.load(request["input"], allow_pickle=False) as arrays:
                    rgb, original_alpha = arrays["rgb"], arrays["original_alpha"]
                if request.get("benchmark"):
                    # Time one photo with the model already loaded.
                    engine.load(progress)
                    started = time.perf_counter()
                    engine.analyze(rgb, original_alpha, lambda _: None, threading.Event())
                    emit(
                        {
                            "event": "timing",
                            "seconds": time.perf_counter() - started,
                            "mask": getattr(engine, "last_mask_seconds", None),
                        }
                    )
                    continue
                alpha, faces, warnings = engine.analyze(
                    rgb, original_alpha, progress, threading.Event()
                )
                np.save(request["output"], alpha, allow_pickle=False)
                emit({"event": "result", "faces": [asdict(f) for f in faces], "warnings": warnings})
            except Exception as error:  # noqa: BLE001 - marshal model errors back to the editor
                traceback.print_exc(file=sys.stderr)
                emit({"event": "error", "text": str(error)})
    finally:
        engine.close()


if __name__ == "__main__":
    main()
