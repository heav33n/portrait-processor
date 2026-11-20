import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from portrait_processor.ai import Cancelled
from portrait_processor.inference_process import InferenceProcess


@pytest.fixture
def process(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("""
import json, os, sys, time
import numpy as np
for line in sys.stdin:
    request = json.loads(line)
    with np.load(request['input'], allow_pickle=False) as arrays:
        rgb = arrays['rgb']
        mode = int(rgb[0, 0, 0])
        print(json.dumps({'event': 'progress', 'text': 'working'}), flush=True)
        if mode == 0:
            time.sleep(30)
        elif mode == 1:
            os._exit(9)
        np.save(request['output'], np.full(rgb.shape[:2], 255, np.uint8))
        print(json.dumps({'event': 'result', 'faces': [{'nose': [20, 15], 'eye_a': [15, 10], 'eye_b': [25, 10]}], 'warnings': []}), flush=True)
""")
    engine = InferenceProcess(_command=[sys.executable, "-u", str(worker)])
    yield engine
    engine.close()


def arrays(value):
    return np.full((32, 40, 3), value, np.uint8), np.full((32, 40), 255, np.uint8)


def test_reuses_worker_and_cleans_up_transport_files(process):
    rgb, alpha = arrays(200)
    result, faces, warnings = process.analyze(rgb, alpha, lambda _: None, threading.Event())
    assert faces[0].nose == (20, 15)
    assert warnings == []
    np.testing.assert_array_equal(result, alpha)
    pid = process.process.pid
    process.analyze(rgb, alpha, lambda _: None, threading.Event())
    assert process.process.pid == pid
    from pathlib import Path

    assert not (Path(process.directory.name) / "input.npz").exists()
    assert not (Path(process.directory.name) / "output.npy").exists()
    child = process.process
    process.close()
    assert child.poll() is not None


def test_cancel_stops_native_work_and_allows_a_new_worker(process):
    cancel = threading.Event()
    start = time.monotonic()
    with pytest.raises(Cancelled):
        process.analyze(*arrays(0), lambda _: cancel.set(), cancel)
    assert time.monotonic() - start < 5
    assert process.process is None
    result, _, _ = process.analyze(*arrays(200), lambda _: None, threading.Event())
    assert result.min() == 255


def test_worker_crash_returns_control_to_editor(process):
    with pytest.raises(RuntimeError, match="AI process stopped"):
        process.analyze(*arrays(1), lambda _: None, threading.Event())
    assert process.process is None


def test_timeout_stops_worker(process):
    process.timeout = 1.5
    with pytest.raises(TimeoutError, match="Manual setup"):
        process.analyze(*arrays(0), lambda _: None, threading.Event())
    assert process.process is None


def test_vision_import_skips_audio_hardware_initialization():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from portrait_processor.ai import vision_modules; import sys; vision_modules(); assert 'sounddevice' not in sys.modules",
        ],
        check=True,
        timeout=30,
        capture_output=True,
    )


def test_worker_switches_level_per_request_and_times_without_loading(tmp_path, monkeypatch):
    import io
    import json

    from portrait_processor import ai_worker

    calls = []

    class FakeEngine:
        def __init__(self, level):
            self.level = level
            self.last_mask_seconds = 0.25

        def set_level(self, number):
            calls.append(("level", number))
            self.level = number

        def load(self, progress):
            calls.append(("load", self.level))

        def analyze(self, rgb, alpha, progress, cancel):
            calls.append(("analyze", self.level))
            return alpha, [], []

        def close(self):
            pass

    source = tmp_path / "in.npz"
    np.savez(source, rgb=np.zeros((4, 4, 3), np.uint8), original_alpha=np.ones((4, 4), np.uint8))
    requests = [
        {"input": str(source), "output": str(tmp_path / "o.npy"), "level": 3, "benchmark": True},
        {"input": str(source), "output": str(tmp_path / "o.npy"), "level": 1, "benchmark": False},
    ]
    out = io.StringIO()
    monkeypatch.setattr(ai_worker, "InferenceEngine", FakeEngine)
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(r) + "\n" for r in requests)))
    monkeypatch.setattr(sys, "stdout", out)
    ai_worker.main()
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [e["event"] for e in events] == ["timing", "result"]
    assert events[0]["seconds"] >= 0 and events[0]["mask"] == 0.25
    assert calls == [("level", 3), ("load", 3), ("analyze", 3), ("level", 1), ("analyze", 1)]
    assert (tmp_path / "o.npy").is_file()
