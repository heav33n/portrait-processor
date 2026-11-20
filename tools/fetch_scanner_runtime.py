"""Fetch the unmodified, checksum-verified NAPS2 Windows scanning bridge and source."""

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
VERSION = "8.6.0"
SHA256 = "5337339764a948722a7b1d3286116ad486a633c13ec006c81eee3a99b9af12e2"
WORK = ROOT / "artifacts" / "windows-build"
DOWNLOADS = WORK / "downloads"
DOWNLOADS.mkdir(parents=True, exist_ok=True)


def download(url, filename, digest=None):
    target = DOWNLOADS / filename
    if not target.is_file() or (
        digest and hashlib.sha256(target.read_bytes()).hexdigest() != digest
    ):
        temporary = target.with_suffix(".part")
        print(f"Downloading {filename}…", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "PortraitProcessor-build"})
        with (
            urllib.request.urlopen(request, timeout=90) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest and actual != digest:
            temporary.unlink()
            raise ValueError(f"Scanner runtime checksum mismatch: {actual}")
        temporary.replace(target)
    return target


binary_url = (
    f"https://github.com/cyanfish/naps2/releases/download/v{VERSION}/naps2-{VERSION}-win-x64.zip"
)
source_url = f"https://github.com/cyanfish/naps2/archive/refs/tags/v{VERSION}.zip"
binary = download(binary_url, f"naps2-{VERSION}-win-x64.zip", SHA256)
source = download(source_url, f"naps2-{VERSION}-source.zip")
runtime = WORK / "scanner-runtime"
runtime.mkdir(exist_ok=True)
with zipfile.ZipFile(binary) as archive:
    for member in archive.infolist():
        relative = PurePosixPath(member.filename.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid scanner archive path")
        target = runtime.joinpath(*relative.parts)
        if member.is_dir() or member.filename.endswith("\\"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_stream, target.open("wb") as destination:
                shutil.copyfileobj(source_stream, destination)
    # Remove staging files from an earlier extraction of Windows-style ZIP paths.
    for path in runtime.iterdir():
        if "\\" in path.name and path.is_file():
            path.unlink()
sources = runtime / "source"
sources.mkdir(exist_ok=True)
shutil.copy2(source, sources / source.name)
(runtime / "PORTRAIT-PROCESSOR-NOTICE.txt").write_text(
    f"NAPS2 {VERSION}, copyright Ben Olden-Cooligan and contributors.\n"
    "Unmodified portable distribution used as a separate local scanner process.\n"
    "License: GNU GPL 2.0 or later; original licenses/notices are retained.\n"
    "Corresponding NAPS2 source is included in the source folder.\n"
    "Project and build instructions: https://github.com/cyanfish/naps2\n"
    f"Binary source: {binary_url}\nSource archive: {source_url}\n"
    "Scanner drivers are supplied by the device manufacturer, not NAPS2.\n",
    encoding="utf-8",
)
metadata = {
    "version": VERSION,
    "sha256": SHA256,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}
(runtime / "scanner-build.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Verified scanner runtime: {runtime}", flush=True)
print(
    "Console executables:",
    [str(p.relative_to(runtime)) for p in runtime.rglob("NAPS2.Console.exe")],
)
