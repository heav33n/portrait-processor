"""Create review artifacts for a real input folder without changing its images."""

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from portrait_processor.ai import InferenceEngine
from portrait_processor.document import SUPPORTED, Document, render_document, save_png
from portrait_processor.levels import LEVELS

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument(
    "--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "test_batch"
)
parser.add_argument(
    "--level",
    type=int,
    choices=[item.number for item in LEVELS],
    default=1,
    help="AI accuracy level",
)
args = parser.parse_args()
paths = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
output = args.output
for subdir in ("masks", "before_trim", "processed_photos"):
    (output / subdir).mkdir(parents=True, exist_ok=True)
engine = InferenceEngine(level=args.level)
print(f"Level {engine.level}: {engine.model_name}", flush=True)
report = []
for index, source in enumerate(paths):
    print(f"[{index + 1}/{len(paths)}] {source.name}", flush=True)
    started = time.perf_counter()
    doc = Document.load(source)
    alpha, faces, warnings = engine.analyze(
        doc.rgb, doc.original_alpha, lambda text: print(f"  {text}", flush=True), threading.Event()
    )
    doc.apply_analysis(alpha, faces)
    entry: dict[str, Any] = {
        "source": str(source),
        "level": engine.level,
        "model": engine.model_name,
        "faces": len(faces),
        "warnings": warnings,
    }
    Image.fromarray(alpha).save(output / "masks" / f"{source.stem}.png")
    if doc.ready:
        result = render_document(doc)
        assert result["rgb"].shape == (1048, 948, 3)
        assert np.array_equal(result["alpha"], result["before_alpha"])
        assert np.all(result["alpha"] <= result["before_alpha"])
        assert np.any(result["alpha"])
        assert doc.nose is not None
        t = result["transform"]
        nose_x = t[0] * doc.nose[0] + t[1] * doc.nose[1] + t[2]
        assert abs(nose_x - 473.5) < 1e-6
        Image.fromarray(result["before_rgb"]).save(output / "before_trim" / f"{source.stem}.png")
        destination = save_png(result["rgb"], output / "processed_photos", source.stem)
        entry.update(
            {
                "output": str(destination),
                "scale": result["scale"],
                "trim_fraction": result["removed_fraction"],
                "nose_x": nose_x,
                "angle_degrees": doc.angle,
                "nose": doc.nose,
                "eye_a": doc.eye_a,
                "eye_b": doc.eye_b,
            }
        )
    else:
        entry["needs_manual_markers"] = True
    entry["seconds"] = round(time.perf_counter() - started, 2)
    report.append(entry)
    print(f"  Faces: {len(faces)}; completed in {entry['seconds']}s", flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
engine.close()
print(f"Review artifacts: {output}", flush=True)
