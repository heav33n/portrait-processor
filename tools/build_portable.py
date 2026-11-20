"""Build a self-contained Linux tarball or macOS app on the current machine.

Bundles a private Python 3.12, the locked dependencies, the native Rust extension and the
packed models (run tools/pack_models.py first). Nothing needs to be installed on the target.

Usage: uv run --no-sync python tools/build_portable.py [--self-test]
"""

import argparse
import hashlib
import os
import platform
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from portrait_processor import __version__
from portrait_processor.levels import LEVELS

PYTHON = "3.12.15"
MACOS = sys.platform == "darwin"
if sys.platform not in {"linux", "darwin"}:
    raise SystemExit("Use tools/build_windows.py for Windows.")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--self-test", action="store_true", help="Run the bundled self-test after building"
)
args = parser.parse_args()

arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}[
    platform.machine().lower()
]
name = f"PortraitProcessor-{'macOS' if MACOS else 'Linux'}-{arch}"
work = ROOT / "artifacts" / "portable-build"
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
packed_models = ROOT / "artifacts" / "models"
bundled_models = [
    packed_models / f"{model}.fp16.onnx"
    for model in dict.fromkeys(item.model for item in LEVELS if item.bundled)
]
missing = [path.name for path in bundled_models if not path.is_file()]
if missing:
    raise SystemExit(
        f"Missing packed models in {packed_models}: {', '.join(missing)}. "
        "Run `uv run python tools/pack_models.py` first."
    )


def run(*command, env=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


# Layout: Linux → <name>/{PortraitProcessor, runtime, models}
#         macOS → Portrait Processor.app/Contents/{MacOS, Resources/{runtime, models}}
if MACOS:
    top = work / "Portrait Processor.app"
    contents = top / "Contents"
    home = contents / "Resources"
    launcher = contents / "MacOS" / "PortraitProcessor"
else:
    top = work / name
    home = top
    launcher = top / "PortraitProcessor"
runtime = home / "runtime"

run("uv", "python", "install", PYTHON, "--install-dir", work / "python-download", "--no-bin")
source_runtime = next((work / "python-download").glob(f"cpython-{PYTHON}-*"))
shutil.copytree(
    source_runtime,
    runtime,
    symlinks=True,
    ignore=shutil.ignore_patterns("__pycache__", "test", "tests", "idle_test", "include"),
)
python = runtime / "bin" / "python3.12"
packages = runtime / "lib" / "python3.12" / "site-packages"

requirements = work / "requirements.txt"
run(
    "uv", "export", "--frozen", "--no-dev", "--no-emit-project",
    "--format", "requirements-txt", "-o", requirements,
)  # fmt: skip
run(
    "uv", "pip", "install", "--python", python, "--target", packages,
    "--only-binary", ":all:", "--require-hashes", "-r", requirements,
)  # fmt: skip

environment = os.environ.copy()
if MACOS:
    environment.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")
run(
    ROOT / ".venv" / "bin" / "maturin", "build", "--release",
    "--interpreter", python, "--out", work / "wheels", env=environment,
)  # fmt: skip
wheel = next((work / "wheels").glob("portrait_processor-*.whl"))
run("uv", "pip", "install", "--python", python, "--target", packages, "--no-deps", wheel)
shutil.copy2(ROOT / "packaging" / "windows" / "self_test.py", runtime / "self_test.py")

models = home / "models"
models.mkdir()
for path in bundled_models:
    print(f"Bundling {path.name} ({path.stat().st_size / 1024**2:.1f} MB)", flush=True)
    shutil.copy2(path, models / path.name)

launcher.parent.mkdir(parents=True, exist_ok=True)
resolve = 'here="$(cd "$(dirname "$0")" && pwd)"'
if not MACOS:
    # Follow a symlink (for example ~/.local/bin/portrait-processor) back to the folder.
    resolve = 'here="$(dirname "$(readlink -f "$0")")"'
app_home = '"$here/../Resources"' if MACOS else '"$here"'
launcher.write_text(
    "#!/bin/sh\n"
    f"{resolve}\n"
    f"home={app_home}\n"
    'export PORTRAIT_BUNDLED_MODELS="$home/models"\n'
    'if [ "$1" = "--self-test" ]; then\n'
    '  exec "$home/runtime/bin/python3.12" "$home/runtime/self_test.py"\n'
    "fi\n"
    'exec "$home/runtime/bin/python3.12" -m portrait_processor "$@"\n',
    encoding="utf-8",
)
launcher.chmod(0o755)

if MACOS:
    from PIL import Image

    with Image.open(ROOT / "assets" / "portrait-processor.ico") as icon:
        icon.size = (256, 256)
        icon.load()
        icon.save(home / "PortraitProcessor.icns")
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(
            {
                "CFBundleName": "Portrait Processor",
                "CFBundleDisplayName": "Portrait Processor",
                "CFBundleIdentifier": "org.portraitprocessor.PortraitProcessor",
                "CFBundleExecutable": "PortraitProcessor",
                "CFBundleIconFile": "PortraitProcessor.icns",
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": __version__,
                "CFBundleVersion": __version__,
                "LSMinimumSystemVersion": "14.0",
                "NSHighResolutionCapable": True,
            },
            stream,
        )
else:
    shutil.copy2(ROOT / "assets" / "portrait-processor.svg", top / "portrait-processor.svg")
    shortcut = top / "install-shortcut.sh"
    shortcut.write_text(
        "#!/bin/sh\n"
        "# Add Portrait Processor to your application menu.\n"
        'here="$(dirname "$(readlink -f "$0")")"\n'
        'apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"\n'
        'mkdir -p "$apps"\n'
        'cat > "$apps/portrait-processor.desktop" <<EOF\n'
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Portrait Processor\n"
        "Comment=Remove backgrounds and prepare portrait photos\n"
        'Exec="$here/PortraitProcessor" %F\n'
        "Icon=$here/portrait-processor.svg\n"
        "Terminal=false\n"
        "Categories=Graphics;Photography;\n"
        "EOF\n"
        'echo "Added to the application menu: $apps/portrait-processor.desktop"\n',
        encoding="utf-8",
    )
    shortcut.chmod(0o755)

for cache in top.rglob("__pycache__"):
    shutil.rmtree(cache, ignore_errors=True)

if args.self_test:
    test_env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    test_env["PORTRAIT_SELFTEST_OUTPUT"] = str(work / "self-test")
    run(launcher, "--self-test", env=test_env)
    for cache in top.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

dist = ROOT / "dist"
dist.mkdir(exist_ok=True)
if MACOS:
    destination = dist / f"{name}.zip"
    destination.unlink(missing_ok=True)
    run("ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", top, destination)
else:
    destination = dist / f"{name}.tar.gz"
    run("tar", "-C", work, "-czf", destination, name)
with destination.open("rb") as stream:
    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
Path(f"{destination}.sha256").write_text(f"{checksum}  {destination.name}\n", encoding="utf-8")
print(f"Package: {destination} ({destination.stat().st_size / 1024**2:.1f} MB)", flush=True)
