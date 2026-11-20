"""Download the two pinned model choices before using the desktop editor."""

import hashlib
import os
import threading

from portrait_processor.ai import face_model

os.environ.setdefault("OMP_NUM_THREADS", "6")
path = face_model(lambda text: print(text, flush=True), threading.Event())
print(f"Face model: {path}", flush=True)
print(f"SHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
print("Preparing BiRefNet Portrait (first use downloads its ONNX weights)…", flush=True)
from rembg import new_session

session = new_session("birefnet-portrait", providers=["CPUExecutionProvider"])
print("Both models are ready for local processing.", flush=True)
