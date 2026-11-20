import threading

import pytest
from portrait_processor import ai


def test_failed_model_download_removes_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ai, "model_directory", lambda: tmp_path)

    def unavailable(*args, **kwargs):
        raise OSError("No network")

    monkeypatch.setattr("portrait_processor.ai.urllib.request.urlopen", unavailable)
    with pytest.raises(OSError, match="No network"):
        ai.face_model(lambda text: None, threading.Event())
    assert list(tmp_path.iterdir()) == []


def test_cancelled_model_operation_stops_before_loading():
    event = threading.Event()
    event.set()
    with pytest.raises(ai.Cancelled):
        ai.check_cancel(event)
