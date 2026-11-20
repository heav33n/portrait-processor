"""Bundled background models, stored at half precision and expanded once on first use.

Only the weights change: float32 initializers are stored as float16 and their names are
recorded, so expansion casts exactly those back. The graph itself is never rewritten.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .i18n import tr

LOG = logging.getLogger(__name__)
PACKED_KEY = "portrait_processor.float16_initializers"
FLOAT16_LIMIT = 65000.0


def pack_model(source: Path, target: Path) -> None:
    import onnx
    from onnx import TensorProto, numpy_helper

    model = onnx.load(str(source))
    packed = []
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.data_type != TensorProto.FLOAT:
            continue
        values = numpy_helper.to_array(tensor)
        if values.size and float(np.max(np.abs(values))) > FLOAT16_LIMIT:
            continue
        model.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(values.astype(np.float16), tensor.name)
        )
        packed.append(tensor.name)
    entry = model.metadata_props.add()
    entry.key, entry.value = PACKED_KEY, "\n".join(packed)
    onnx.save(model, str(target))


def expand_model(source: Path, target: Path) -> None:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(source))
    entries = [p for p in model.metadata_props if p.key == PACKED_KEY]
    if not entries:
        raise ValueError(tr("Not a Portrait Processor half-precision model."))
    names = set(filter(None, entries[0].value.split("\n")))
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.name in names:
            values = numpy_helper.to_array(tensor).astype(np.float32)
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(values, tensor.name))
    kept = [p for p in model.metadata_props if p.key != PACKED_KEY]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    onnx.save(model, str(target))


def bundled_directory() -> Path | None:
    override = os.environ.get("PORTRAIT_BUNDLED_MODELS")
    if override:
        return Path(override)
    if sys.platform == "win32":
        # <app>/runtime/python.exe → <app>/models
        return Path(sys.executable).resolve().parent.parent / "models"
    return None


def prepare_model(
    model: str,
    rembg_home: Path,
    progress: Callable[[str], None],
    bundle: Path | None = None,
) -> None:
    """Expand a bundled model into rembg's cache layout; otherwise leave rembg to download."""
    bundle = bundle if bundle is not None else bundled_directory()
    target = Path(rembg_home) / "models" / model / f"{model}.onnx"
    if target.is_file() or bundle is None:
        return
    source = Path(bundle) / f"{model}.fp16.onnx"
    if not source.is_file():
        return
    progress(tr("Preparing the bundled model (first use only)…"))
    target.parent.mkdir(parents=True, exist_ok=True)
    # A per-process name keeps two running copies from writing the same temporary file.
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        expand_model(source, temporary)
        _replace_when_unlocked(temporary, target)
    except Exception:
        LOG.warning("Could not expand bundled model %s", model, exc_info=True)
        for cleanup in (lambda: temporary.unlink(missing_ok=True), target.parent.rmdir):
            try:
                cleanup()
            except OSError:
                pass


def _replace_when_unlocked(source: Path, target: Path, attempts: int = 8) -> None:
    """Windows antivirus scanners briefly hold freshly written files open; wait them out."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))
