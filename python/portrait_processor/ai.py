"""Lazy, CPU-based model inference. No photo is uploaded."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from . import _native
from .document import Face
from .i18n import tr
from .levels import display_name
from .levels import level as accuracy_level

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
# SHA-256 of Google's official version 1 float16 asset.
FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


class Cancelled(Exception):
    pass


def vision_modules():
    """MediaPipe eagerly imports audio; avoid enumerating unused audio hardware."""
    blocked = "sounddevice" not in sys.modules
    if blocked:
        sys.modules["sounddevice"] = None  # pyright: ignore[reportArgumentType]
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        return mp, python, vision
    finally:
        if blocked:
            sys.modules.pop("sounddevice", None)


def check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled(tr("Processing cancelled."))


def model_directory() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "PortraitProcessor" / "Cache" / "models"
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "portrait-processor" / "models"


def face_model(progress: Callable[[str], None], cancel: threading.Event) -> Path:
    directory = model_directory()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "face_landmarker.task"
    if target.is_file() and (
        not FACE_MODEL_SHA256
        or hashlib.sha256(target.read_bytes()).hexdigest() == FACE_MODEL_SHA256
    ):
        return target
    progress(tr("Downloading face landmarks model…"))
    fd, temporary = tempfile.mkstemp(prefix="face-", suffix=".part", dir=directory)
    try:
        with (
            os.fdopen(fd, "wb") as output,
            urllib.request.urlopen(FACE_MODEL_URL, timeout=45) as response,
        ):
            while block := response.read(1024 * 256):
                check_cancel(cancel)
                output.write(block)
        downloaded = Path(temporary)
        if (
            FACE_MODEL_SHA256
            and hashlib.sha256(downloaded.read_bytes()).hexdigest() != FACE_MODEL_SHA256
        ):
            raise RuntimeError(tr("Face model checksum failed. Retry the download."))
        downloaded.replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def refine_edges(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Snap a low-resolution mask's soft outline to the edges of the full-size photo.

    A guided filter fits the mask to the photo locally; only the uncertain band around the
    outline changes, so solid subject and clear background keep their values.
    """
    import cv2

    h, w = alpha.shape
    radius = max(4, round(min(h, w) / 160))
    band = (alpha > 8) & (alpha < 247)
    if not band.any():
        return alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    band = cv2.dilate(band.astype(np.uint8), kernel) > 0
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    source = alpha.astype(np.float32) / 255
    fitted = cv2.ximgproc.guidedFilter(guide, source, radius, 1e-4)
    # The fit follows the photo's edge; a gentle contrast pulls the ramp to that edge.
    fitted = np.clip((fitted - 0.5) * 2.0 + 0.5, 0, 1)
    result = alpha.copy()
    result[band] = np.round(fitted[band] * 255).astype(np.uint8)
    return result


class InferenceEngine:
    def __init__(self, level: int = 1) -> None:
        self.level = level
        self.session = None
        self.landmarker = None
        self.last_mask_seconds = None

    @property
    def model_name(self) -> str:
        return accuracy_level(self.level).model

    def set_level(self, number: int) -> None:
        """Switch level; the loaded model is kept when both levels share it."""
        if accuracy_level(number).model != self.model_name:
            self.session = None
        self.level = number

    def load(self, progress: Callable[[str], None]) -> None:
        if sys.platform == "win32":
            os.environ.setdefault("REMBG_HOME", str(model_directory() / "background"))
        # rembg honors this when constructing its ONNX Runtime session.
        os.environ.setdefault("OMP_NUM_THREADS", str(min(6, os.cpu_count() or 2)))
        if self.session is not None:
            return
        from rembg import new_session

        settings = accuracy_level(self.level)
        if os.environ.get("REMBG_HOME"):
            from .models import prepare_model

            prepare_model(settings.model, Path(os.environ["REMBG_HOME"]), progress)
        name = display_name(settings)
        progress(
            tr("Loading the {name} portrait model…", name=name)
            if settings.bundled
            else tr("Loading the {name} portrait model (first use downloads weights)…", name=name)
        )
        import onnxruntime as ort

        options = ort.SessionOptions()
        # Avoid retaining large activation arenas between portraits on smaller PCs.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = new_session(
            settings.model, sess_opts=options, providers=["CPUExecutionProvider"]
        )

    def analyze(
        self,
        rgb: np.ndarray,
        original_alpha: np.ndarray,
        progress: Callable[[str], None],
        cancel: threading.Event,
    ) -> tuple[np.ndarray, list[Face], list[str]]:
        check_cancel(cancel)
        self.load(progress)
        from rembg import remove

        check_cancel(cancel)
        progress(tr("Separating the portrait from its background…"))
        started = time.perf_counter()
        mask = remove(Image.fromarray(rgb), session=self.session, only_mask=True)
        self.last_mask_seconds = time.perf_counter() - started
        assert isinstance(mask, Image.Image)
        alpha = np.asarray(mask.convert("L"))
        if accuracy_level(self.level).refine:
            progress(tr("Refining the portrait edges…"))
            alpha = refine_edges(rgb, alpha)
        alpha = np.minimum(alpha, original_alpha).astype(np.uint8)
        check_cancel(cancel)
        warnings = []
        try:
            faces = self.detect_faces(rgb, progress, cancel)
        except Cancelled:
            raise
        except Exception as error:
            logging.getLogger(__name__).warning("Landmark detection failed", exc_info=True)
            faces = []
            warnings.append(
                tr(
                    "Automatic landmarks unavailable: {error}. Set the nose and eyes manually.",
                    error=error,
                )
            )
        check_cancel(cancel)
        anchor = faces[0].nose if len(faces) == 1 else None
        progress(tr("Cleaning small background particles…"))
        alpha = np.asarray(_native.clean_specks(alpha, 24, anchor))
        return alpha, faces, warnings

    def detect_faces(
        self, rgb: np.ndarray, progress: Callable[[str], None], cancel: threading.Event
    ) -> list[Face]:
        progress(tr("Loading facial landmark detector…"))
        mp, python, vision = vision_modules()

        if self.landmarker is None:
            path = face_model(progress, cancel)
            self.landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(path), delegate=python.BaseOptions.Delegate.CPU
                    ),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=6,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                )
            )
        check_cancel(cancel)
        progress(tr("Finding the nose and eye positions…"))
        image = Image.fromarray(rgb)
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        result = self.landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
        )
        h, w = rgb.shape[:2]

        def point(landmarks, indices):
            x = sum(landmarks[i].x for i in indices) / len(indices) * (w - 1)
            y = sum(landmarks[i].y for i in indices) / len(indices) * (h - 1)
            return float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))

        faces = [
            Face(point(lm, [1]), point(lm, [33, 133]), point(lm, [263, 362]))
            for lm in result.face_landmarks
        ]
        return sorted(faces, key=lambda face: face.nose[0])

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        self.session = None
