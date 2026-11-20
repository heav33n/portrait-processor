"""Real inference smoke test using scikit-image's public-domain astronaut photo."""

import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from portrait_processor.ai import InferenceEngine
from portrait_processor.document import Document, render_document, save_png
from skimage import data

output = Path(__file__).resolve().parents[1] / "artifacts"
output.mkdir(exist_ok=True)
rgb = data.astronaut()
source = output / "astronaut_source.png"
Image.fromarray(rgb).save(source)
engine = InferenceEngine(level=1)
print(f"Level {engine.level}: {engine.model_name}", flush=True)
started = time.perf_counter()
doc = Document.load(source)
alpha, faces, warnings = engine.analyze(
    doc.rgb, doc.original_alpha, lambda text: print(text, flush=True), threading.Event()
)
assert faces, f"Expected a detectable face: {warnings}"
doc.apply_analysis(alpha, faces)
if len(faces) > 1:
    doc.choose_face(0)
result = render_document(doc)
assert result["rgb"].shape == (1048, 948, 3)
assert np.array_equal(result["alpha"], result["before_alpha"])
assert np.all(result["alpha"] <= result["before_alpha"])
assert np.count_nonzero(alpha) > 10000, "The subject was lost"
assert np.count_nonzero(alpha == 0) > 10000, "The background was not removed"
Image.fromarray(alpha).save(output / "astronaut_mask.png")
Image.fromarray(result["before_rgb"]).save(output / "astronaut_before_trim.png")
destination = save_png(result["rgb"], output, "astronaut_processed")
print(f"Faces: {len(faces)}; warnings: {warnings}", flush=True)
print(f"Foreground trimmed: {result['removed_fraction']:.1%}", flush=True)
print(f"Complete in {time.perf_counter() - started:.1f}s: {destination}", flush=True)
engine.close()
