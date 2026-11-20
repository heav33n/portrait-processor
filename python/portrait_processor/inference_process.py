"""Keep native model loading and inference outside Qt, with interruptible IPC."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .ai import check_cancel
from .document import Face
from .i18n import tr
from .levels import level as accuracy_level

LOG = logging.getLogger(__name__)


class InferenceProcess:
    def __init__(self, *, timeout=900, level=1, _command=None):
        self.level = level
        self.timeout = timeout
        executable = (
            str(Path(sys.executable).with_name("python.exe"))
            if sys.platform == "win32"
            else sys.executable
        )
        self.command = _command or [executable, "-u", "-m", "portrait_processor.ai_worker"]
        self.process = None
        self.directory = None
        self.messages = None
        self.log = None
        self.lock = threading.Lock()

    @property
    def model_name(self):
        return accuracy_level(self.level).model

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self._stop()
        self.directory = tempfile.TemporaryDirectory(prefix="portrait-ai-")
        # Keep this handle open for the persistent child's entire lifetime.
        self.log = open(Path(self.directory.name) / "worker.log", "w", encoding="utf-8")  # noqa: SIM115
        environment = os.environ.copy()
        environment.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 2)))
        environment["PORTRAIT_AI_LEVEL"] = str(self.level)
        if self.model_name == "u2net_human_seg":
            environment["OMP_NUM_THREADS"] = str(min(2, int(environment["OMP_NUM_THREADS"])))
        flags = (
            subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            if sys.platform == "win32"
            else 0
        )
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            env=environment,
            creationflags=flags,
        )
        self.messages = queue.Queue()
        messages = self.messages
        stdout = self.process.stdout
        assert stdout is not None

        def read():
            try:
                for line in stdout:
                    try:
                        messages.put(json.loads(line))
                    except ValueError:
                        LOG.debug("AI output: %s", line.rstrip())
            finally:
                messages.put({"event": "exit"})

        threading.Thread(target=read, name="portrait-ai-output", daemon=True).start()

    def _stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    def analyze(self, rgb, original_alpha, progress, cancel):
        return self._request(rgb, original_alpha, progress, cancel, self.level, False)

    def measure_level(self, level, rgb, original_alpha, progress, cancel):
        """Seconds one photo takes at `level` excluding model loading, as (total, background
        model share) when the worker reports the share."""
        return self._request(rgb, original_alpha, progress, cancel, level, True)

    def _request(self, rgb, original_alpha, progress, cancel, level, benchmark):
        with self.lock:
            check_cancel(cancel)
            try:
                self._start()
                assert self.directory is not None and self.messages is not None
                assert self.process is not None and self.process.stdin is not None
                source = Path(self.directory.name) / "input.npz"
                target = Path(self.directory.name) / "output.npy"
                np.savez(source, rgb=rgb, original_alpha=original_alpha)
                request = {
                    "input": str(source),
                    "output": str(target),
                    "level": level,
                    "benchmark": benchmark,
                }
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                started = time.monotonic()
                while True:
                    check_cancel(cancel)
                    if time.monotonic() - started > self.timeout:
                        raise TimeoutError(tr("AI took too long. Retry AI or use Manual setup."))
                    try:
                        message = self.messages.get(timeout=0.15)
                    except queue.Empty:
                        continue
                    kind = message.get("event")
                    if kind == "progress":
                        progress(message["text"])
                    elif kind == "timing":
                        seconds = float(message["seconds"])
                        mask = message.get("mask")
                        return seconds if mask is None else (seconds, float(mask))
                    elif kind == "result":
                        alpha = np.load(target, allow_pickle=False)
                        if alpha.shape != original_alpha.shape or alpha.dtype != np.uint8:
                            raise RuntimeError(
                                tr("AI returned an invalid mask. Retry AI or use Manual setup.")
                            )
                        faces = [
                            Face(**{k: tuple(v) for k, v in face.items()})
                            for face in message["faces"]
                        ]
                        return alpha, faces, message["warnings"]
                    elif kind == "error":
                        raise RuntimeError(message["text"])
                    elif kind == "exit":
                        raise RuntimeError(
                            tr("The AI process stopped. Retry AI or use Manual setup.")
                        )
            except Exception:
                if self.directory is not None:
                    try:
                        details = (Path(self.directory.name) / "worker.log").read_text(
                            encoding="utf-8"
                        )[-3000:]
                        if details:
                            LOG.info("AI worker log: %s", details)
                    except OSError:
                        pass
                self._stop()
                raise
            finally:
                if self.directory is not None:
                    for filename in ("input.npz", "output.npy"):
                        (Path(self.directory.name) / filename).unlink(missing_ok=True)

    def close(self):
        with self.lock:
            self._stop()
