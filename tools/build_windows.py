"""Build a portable Windows x64 ZIP on Linux using MinGW and uv's private runtime."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from portrait_processor.levels import LEVELS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--skip-build", action="store_true", help="Reuse the previously built native wheel"
)
parser.add_argument("--skip-deps", action="store_true", help="Reuse downloaded Windows libraries")
args = parser.parse_args()
work = ROOT / "artifacts" / "windows-build"
portable = work / "portable"
runtime = portable / "runtime"
packages = runtime / "Lib" / "site-packages"
runtime.mkdir(parents=True, exist_ok=True)
# Half-precision background models written by tools/pack_models.py; checked before the long build.
packed_models = ROOT / "artifacts" / "models"
bundled_models = [
    packed_models / f"{model}.fp16.onnx"
    for model in dict.fromkeys(item.model for item in LEVELS if item.bundled)
]
missing_models = [path.name for path in bundled_models if not path.is_file()]
if missing_models:
    raise SystemExit(
        f"Missing packed models in {packed_models}: {', '.join(missing_models)}. "
        "Run `uv run python tools/pack_models.py` first."
    )


def run(*command, env=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


run(
    "uv",
    "python",
    "install",
    "cpython-3.12.15-windows-x86_64-none",
    "--install-dir",
    work / "python-download",
    "--no-bin",
    "--no-registry",
)
python_runtime = work / "python-download" / "cpython-3.12.15-windows-x86_64-none"
shutil.copytree(
    python_runtime,
    runtime,
    dirs_exist_ok=True,
    ignore=shutil.ignore_patterns(
        "site-packages", "__pycache__", "include", "Include", "libs", "Scripts", "python"
    ),
)
(runtime / "python312._pth").write_text(
    ".\nDLLs\nLib\nLib/site-packages\nimport site\n", encoding="utf-8"
)
if not args.skip_deps:
    run(
        "uv",
        "pip",
        "install",
        "--target",
        packages,
        "--python-platform",
        "windows",
        "--python-version",
        "3.12",
        "--only-binary",
        ":all:",
        "--require-hashes",
        "-r",
        ROOT / "packaging" / "windows" / "requirements.lock",
    )
if not args.skip_build:
    environment = os.environ.copy()
    environment.update(CARGO_BUILD_JOBS="2", PYO3_CROSS="1", PYO3_CROSS_PYTHON_VERSION="3.12")
    run(
        ROOT / ".venv" / "bin" / "maturin",
        "build",
        "--release",
        "--target",
        "x86_64-pc-windows-gnu",
        "--interpreter",
        "python3.12",
        "--features",
        "pyo3/generate-import-lib",
        "--out",
        work / "wheels",
        env=environment,
    )
wheel = max((work / "wheels").glob("portrait_processor-*-win_amd64.whl"))
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(packages)
# Copy the latest Python UI alongside the compiled extension; never copy Linux binaries.
shutil.copytree(
    ROOT / "python" / "portrait_processor",
    packages / "portrait_processor",
    dirs_exist_ok=True,
    ignore=shutil.ignore_patterns("*.so", "*.pyd", "__pycache__"),
)
for dll in (packages / "PySide6").glob("*.dll"):
    if dll.name.lower().startswith(("msvcp", "vcruntime", "concrt")):
        shutil.copy2(dll, runtime / dll.name)
for name in ("start_app.py", "self_test.py"):
    shutil.copy2(ROOT / "packaging" / "windows" / name, runtime / name)
shutil.copy2(ROOT / "packaging" / "windows" / "README.txt", portable / "README.txt")
scanner = work / "scanner-runtime"
if not (scanner / "App" / "NAPS2.Console.exe").is_file():
    raise SystemExit("Run tools/fetch_scanner_runtime.py before building scanner support.")
shutil.copytree(scanner, portable / "scanner", dirs_exist_ok=True)
models = portable / "models"
shutil.rmtree(models, ignore_errors=True)
models.mkdir()
for path in bundled_models:
    print(f"Bundling {path.name} ({path.stat().st_size / 1024**2:.1f} MB)", flush=True)
    shutil.copy2(path, models / path.name)
resource = work / "launcher.rc"
manifest = (ROOT / "packaging" / "windows" / "launcher.manifest").as_posix()
icon = (ROOT / "assets" / "portrait-processor.ico").as_posix()
resource.write_text(f'1 24 "{manifest}"\n1 ICON "{icon}"\n', encoding="utf-8")
run("x86_64-w64-mingw32-windres", resource, "-O", "coff", "-o", work / "launcher.res")
run(
    "x86_64-w64-mingw32-gcc",
    "-municode",
    "-mwindows",
    "-O2",
    "-static-libgcc",
    ROOT / "packaging" / "windows" / "launcher.c",
    work / "launcher.res",
    "-o",
    portable / "PortraitProcessor.exe",
    "-luser32",
)
metadata = {
    "python": "3.12.15",
    "architecture": "Windows x64",
    "native_wheel": wheel.name,
    "native_sha256": hashlib.sha256(
        next((packages / "portrait_processor").glob("*.pyd")).read_bytes()
    ).hexdigest(),
}
(portable / "BUILDINFO.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
destination = ROOT / "dist" / "PortraitProcessor-Windows-x64.zip"
destination.parent.mkdir(exist_ok=True)
print("Creating portable ZIP…", flush=True)
st_time = time.perf_counter()
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(portable.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            archive.write(path, Path("PortraitProcessor") / path.relative_to(portable))
with destination.open("rb") as stream:
    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
destination.with_suffix(".zip.sha256").write_text(
    f"{checksum}  {destination.name}\n", encoding="utf-8"
)
print(
    f"Windows package: {destination} ({destination.stat().st_size / 1024**2:.1f} MB), took: {int(time.perf_counter() - st_time)} seconds",
    flush=True,
)
