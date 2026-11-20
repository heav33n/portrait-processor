"""Image state, reversible source-mask edits, and collision-safe PNG exports."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import _native
from .i18n import tr

WIDTH, HEIGHT = 948, 1048
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
EXPORT_FORMATS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "TIFF": "tiff", "BMP": "bmp"}
EXPORT_SUFFIXES = {
    "png": "PNG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "webp": "WEBP",
    "tif": "TIFF",
    "tiff": "TIFF",
    "bmp": "BMP",
}
Point = tuple[float, float]


@dataclass
class Face:
    nose: Point
    eye_a: Point
    eye_b: Point


@dataclass
class Settings:
    headroom: int = 180
    angle_offset: float = 0.0
    side_margin: int = 64
    bottom_margin: int = 0
    strict_symmetry: bool = False
    horizontal_offset: int = 0
    position_x: int = 0
    position_y: int = 0


@dataclass
class MaskChange:
    box: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray


@dataclass
class Document:
    path: Path
    rgb: np.ndarray
    original_alpha: np.ndarray
    alpha: np.ndarray
    base_alpha: np.ndarray
    settings: Settings = field(default_factory=Settings)
    faces: list[Face] = field(default_factory=list)
    nose: Point | None = None
    eye_a: Point | None = None
    eye_b: Point | None = None
    analyzed: bool = False
    manual_placement: bool = False
    dirty: bool = False
    revision: int = 0
    result_revision: int = -1
    result: dict | None = None
    saved_path: Path | None = None
    output_name: str | None = None
    export_format: str = "PNG"
    export_quality: int = 95
    undo_stack: list[MaskChange] = field(default_factory=list)
    redo_stack: list[MaskChange] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        path: Path,
        headroom: int = 180,
        *,
        side_margin: int = 64,
        bottom_margin: int = 0,
        strict_symmetry: bool = False,
        horizontal_offset: int = 0,
    ) -> Document:
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source)
            rgba = np.array(oriented.convert("RGBA"), dtype=np.uint8)
        rgb = np.ascontiguousarray(rgba[:, :, :3])
        alpha = np.ascontiguousarray(rgba[:, :, 3])
        return cls(
            path.resolve(),
            rgb,
            alpha.copy(),
            alpha.copy(),
            alpha.copy(),
            settings=Settings(
                headroom=headroom,
                side_margin=side_margin,
                bottom_margin=bottom_margin,
                strict_symmetry=strict_symmetry,
                horizontal_offset=horizontal_offset,
            ),
        )

    @property
    def ready(self) -> bool:
        return self.manual_placement or (
            self.analyzed and all(p is not None for p in (self.nose, self.eye_a, self.eye_b))
        )

    @property
    def angle(self) -> float:
        if self.eye_a is None or self.eye_b is None:
            return self.settings.angle_offset
        a, b = sorted((self.eye_a, self.eye_b))
        dx, dy = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dy) < 1.0:
            raise ValueError(tr("Place the two eye markers at different positions."))
        return -math.degrees(math.atan2(dy, dx)) + self.settings.angle_offset

    def changed(self) -> None:
        self.revision += 1
        self.dirty = True

    def choose_face(self, index: int) -> None:
        face = self.faces[index]
        self.nose, self.eye_a, self.eye_b = face.nose, face.eye_a, face.eye_b
        self.changed()

    def apply_analysis(self, alpha: np.ndarray, faces: list[Face]) -> None:
        self.manual_placement = False
        self.alpha = np.minimum(alpha, self.original_alpha).astype(np.uint8)
        self.base_alpha = self.alpha.copy()
        self.faces = faces
        self.nose = self.eye_a = self.eye_b = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.analyzed = True
        if len(faces) == 1:
            self.choose_face(0)
        else:
            self.changed()

    def record_mask(self, before: np.ndarray) -> None:
        ys, xs = np.nonzero(before != self.alpha)
        if xs.size == 0:
            return
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        box = (x0, y0, x1, y1)
        self.undo_stack.append(
            MaskChange(box, before[y0:y1, x0:x1].copy(), self.alpha[y0:y1, x0:x1].copy())
        )
        # Keep history bounded even for brushes over very large photographs.
        while len(self.undo_stack) > 1 and (
            len(self.undo_stack) > 30
            or sum(c.before.nbytes + c.after.nbytes for c in self.undo_stack) > 96 * 1024**2
        ):
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.changed()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        change = self.undo_stack.pop()
        x0, y0, x1, y1 = change.box
        self.alpha[y0:y1, x0:x1] = change.before
        self.redo_stack.append(change)
        self.changed()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        change = self.redo_stack.pop()
        x0, y0, x1, y1 = change.box
        self.alpha[y0:y1, x0:x1] = change.after
        self.undo_stack.append(change)
        self.changed()

    def paint(self, point: Point, diameter: int, restore: bool) -> None:
        x, y = point
        radius = diameter / 2
        h, w = self.alpha.shape
        x0, x1 = max(0, int(x - radius - 1)), min(w, int(x + radius + 2))
        y0, y1 = max(0, int(y - radius - 1)), min(h, int(y + radius + 2))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        coverage = np.clip((radius - distance) / max(1.0, radius * 0.15), 0, 1)
        current = self.alpha[y0:y1, x0:x1]
        target = self.original_alpha[y0:y1, x0:x1] if restore else 0
        current[:] = np.rint(current * (1 - coverage) + target * coverage).astype(np.uint8)

    def reset(self) -> None:
        self.alpha = self.base_alpha.copy()
        self.settings = Settings()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.nose = self.eye_a = self.eye_b = None
        if len(self.faces) == 1:
            face = self.faces[0]
            self.nose, self.eye_a, self.eye_b = face.nose, face.eye_a, face.eye_b
        self.changed()

    def stash(self, path: Path) -> None:
        """Keep a temporary queue draft while releasing full-resolution RGB memory."""
        metadata = {
            "settings": asdict(self.settings),
            "faces": [asdict(face) for face in self.faces],
            "nose": self.nose,
            "eye_a": self.eye_a,
            "eye_b": self.eye_b,
            "analyzed": self.analyzed,
            "manual_placement": self.manual_placement,
            "dirty": self.dirty,
            "revision": self.revision,
            "saved_path": str(self.saved_path) if self.saved_path else None,
            "output_name": self.output_name,
            "export_format": self.export_format,
            "export_quality": self.export_quality,
        }
        arrays = {"alpha": self.alpha, "base_alpha": self.base_alpha}
        for name, stack in [("undo", self.undo_stack), ("redo", self.redo_stack)]:
            metadata[name] = [change.box for change in stack]
            for i, change in enumerate(stack):
                arrays[f"{name}_{i}_before"] = change.before
                arrays[f"{name}_{i}_after"] = change.after
        np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)

    @classmethod
    def from_stash(cls, source: Path, path: Path) -> Document:
        doc = cls.load(source)
        with np.load(path, allow_pickle=False) as draft:
            info = json.loads(str(draft["metadata"]))
            if draft["alpha"].shape != doc.original_alpha.shape:
                raise ValueError(tr("The source image dimensions changed since it was opened."))
            doc.alpha = draft["alpha"].copy()
            doc.base_alpha = draft["base_alpha"].copy()
            doc.settings = Settings(**info["settings"])
            doc.faces = [
                Face(**{key: tuple(value) for key, value in face.items()}) for face in info["faces"]
            ]
            for name in ("nose", "eye_a", "eye_b"):
                value = info[name]
                setattr(doc, name, tuple(value) if value is not None else None)
            doc.analyzed = info["analyzed"]
            doc.manual_placement = info.get("manual_placement", False)
            doc.dirty = info["dirty"]
            doc.revision = info["revision"]
            doc.saved_path = Path(info["saved_path"]) if info["saved_path"] else None
            doc.output_name = info.get("output_name")
            doc.export_format = info.get("export_format", "PNG")
            doc.export_quality = info.get("export_quality", 95)
            for name in ("undo", "redo"):
                stack = [
                    MaskChange(
                        tuple(box),
                        draft[f"{name}_{i}_before"].copy(),
                        draft[f"{name}_{i}_after"].copy(),
                    )
                    for i, box in enumerate(info[name])
                ]
                setattr(doc, f"{name}_stack", stack)
        return doc


def render_document(doc: Document) -> dict:
    if not doc.ready:
        raise ValueError(tr("Set the nose and both eye markers before rendering."))
    nose = doc.nose or ((doc.rgb.shape[1] - 1) / 2, (doc.rgb.shape[0] - 1) / 2)
    result = _native.render(
        doc.rgb,
        doc.alpha,
        nose,
        doc.angle,
        doc.settings.headroom,
        doc.settings.side_margin,
        doc.settings.bottom_margin,
        doc.settings.strict_symmetry,
        doc.settings.horizontal_offset,
    )
    dx, dy = doc.settings.position_x, doc.settings.position_y
    if dx or dy:
        for name in ("rgb", "before_rgb", "alpha", "before_alpha"):
            result[name] = translate_layer(result[name], dx, dy, 255 if "rgb" in name else 0)
        t = list(result["transform"])
        t[2] += dx
        t[5] += dy
        result["transform"] = t
        a, b, x, d, e, y = t
        det = a * e - b * d
        result["inverse"] = [
            e / det,
            -b / det,
            (b * y - e * x) / det,
            -d / det,
            a / det,
            (d * x - a * y) / det,
        ]
    return result


def translate_layer(array, dx, dy, fill):
    """Translate whole pixels without resampling or changing the portrait's size."""
    h, w = array.shape[:2]
    out = np.full_like(array, fill)
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = array[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


def export_stem(name: str) -> str:
    name = name.strip()
    if name.rsplit(".", 1)[-1].lower() in EXPORT_SUFFIXES and "." in name:
        name = name.rsplit(".", 1)[0].rstrip()
    if not name or name in {".", ".."}:
        raise ValueError(tr("Enter a photo name."))
    if any(char in name for char in ("/", "\\")) or any(ord(char) < 32 for char in name):
        raise ValueError(tr("Use a photo name without folders or control characters."))
    if len(name.encode("utf-8")) > 240:
        raise ValueError(tr("The photo name is too long."))
    if sys.platform == "win32":
        validate_windows_name(name)
    return name


def validate_windows_name(name: str) -> None:
    base = name.split(".", 1)[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{digit}" for prefix in ("COM", "LPT") for digit in "123456789¹²³")
    if base in reserved or any(c in name for c in '<>:"|?*') or name.endswith((".", " ")):
        raise ValueError(
            tr("Choose a Windows-compatible photo name without reserved names or characters.")
        )


def save_png(rgb: np.ndarray, directory: Path, stem: str) -> Path:
    return save_image(rgb, directory, stem, "PNG")


def save_image(
    rgb: np.ndarray, directory: Path, stem: str, image_format: str = "PNG", quality: int = 95
) -> Path:
    """Publish a complete photo atomically, without overwriting an existing file."""
    if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8:
        raise ValueError(tr("Export must be a 948×1048 RGB image."))
    stem = export_stem(stem)
    image_format = image_format.upper()
    if image_format not in EXPORT_FORMATS:
        raise ValueError(tr("Choose PNG, JPEG, WebP, TIFF, or BMP."))
    if not 1 <= quality <= 100:
        raise ValueError(tr("Quality must be between 1 and 100."))
    extension = EXPORT_FORMATS[image_format]
    options = {}
    if image_format in {"JPEG", "WEBP"}:
        options["quality"] = quality
    if image_format == "JPEG":
        # Stream high-detail 4:4:4 images; optimized encoding can overflow Pillow's buffer.
        options.update(subsampling=0, optimize=False)
    elif image_format == "TIFF":
        options["compression"] = "tiff_deflate"
    directory.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".portrait-", suffix=f".{extension}", dir=directory)
    temp_path = Path(temporary)
    try:
        with os.fdopen(handle, "wb") as stream:
            Image.fromarray(rgb).save(stream, format=image_format, **options)
            stream.flush()
            os.fsync(stream.fileno())
        index = 1
        while True:
            suffix = "" if index == 1 else f"_{index}"
            destination = directory / f"{stem}{suffix}.{extension}"
            try:
                if sys.platform == "win32":
                    # Windows rename is atomic and fails if the destination exists.
                    # Unlike hard links, this also works on USB drives using exFAT.
                    os.rename(temp_path, destination)
                else:
                    os.link(temp_path, destination)
                return destination
            except FileExistsError:
                index += 1
    finally:
        temp_path.unlink(missing_ok=True)
