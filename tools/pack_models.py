"""Download background models, time them with two threads, and pack the bundled ones.

Usage: uv run python tools/pack_models.py --out artifacts/models [--home artifacts/rembg-home]
       [--image portrait.png] [--skip-timing] [--levels 1,3,4]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from portrait_processor.levels import LEVELS
from portrait_processor.models import expand_model, pack_model

SAMPLE_URL = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png"
)


def session(model, home, path=None):
    import onnxruntime as ort
    from rembg import new_session

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    os.environ["REMBG_HOME"] = str(home)
    result = new_session(model, sess_opts=options, providers=["CPUExecutionProvider"])
    if path is not None:
        result.inner_session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    return result


def mask(sess, image):
    from rembg import remove

    return np.asarray(remove(image, session=sess, only_mask=True).convert("L"), np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "models")
    parser.add_argument("--home", type=Path, default=ROOT / "artifacts" / "rembg-home")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--levels", default=",".join(str(lv.number) for lv in LEVELS))
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    image_path = args.image or args.home / "astronaut.png"
    if not image_path.is_file():
        args.home.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, image_path)
    image = Image.open(image_path).convert("RGB").resize((1500, 1500), Image.Resampling.LANCZOS)
    wanted = {int(n) for n in args.levels.split(",")}
    report = {}
    done = set()
    for level in LEVELS:
        if level.number not in wanted or level.model in done:
            continue
        done.add(level.model)
        print(f"== {level.model}", flush=True)
        original_session = session(level.model, args.home)
        original = args.home / "models" / level.model / f"{level.model}.onnx"
        entry = {"size_mb": round(original.stat().st_size / 1e6, 1)}
        reference = mask(original_session, image)
        if not args.skip_timing:
            runs = []
            for _ in range(3):
                started = time.perf_counter()
                mask(original_session, image)
                runs.append(time.perf_counter() - started)
            entry["seconds"] = round(statistics.median(runs), 2)
        if level.bundled:
            packed = args.out / f"{level.model}.fp16.onnx"
            pack_model(original, packed)
            expanded = args.home / f"{level.model}.expanded.onnx"
            expand_model(packed, expanded)
            difference = np.abs(mask(session(level.model, args.home, expanded), image) - reference)
            expanded.unlink()
            entry.update(
                packed_mb=round(packed.stat().st_size / 1e6, 1),
                mask_mean_diff=round(float(difference.mean()), 4),
                mask_max_diff=float(difference.max()),
            )
            if entry["mask_mean_diff"] >= 1.0:
                raise SystemExit(f"{level.model}: half-precision mask differs too much: {entry}")
        print(json.dumps(entry), flush=True)
        report[level.model] = entry
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
