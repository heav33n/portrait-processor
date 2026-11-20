"""Direct flatbed acquisition through a separate, bundled NAPS2 scanner process."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .ai import check_cancel, model_directory
from .document import export_stem
from .i18n import tr


@dataclass(frozen=True)
class ScannerSettings:
    device: str
    driver: str = "twain"
    dpi: int = 1200
    scale: int = 100
    page: str = "216x297mm"
    mode: str = "color"

    @property
    def output_dpi(self):
        return self.dpi * self.scale / 100

    @property
    def acquisition_dpi(self):
        """Scan at the lowest supported resolution that still covers the output size.

        Like Canon ScanGear's output-size setting: 1200 DPI at 49% reads the glass at
        600 DPI instead of capturing four times the pixels and discarding them.
        """
        return next(dpi for dpi in (150, 300, 600, 1200) if dpi >= min(self.output_dpi, self.dpi))

    def validate(self):
        if not self.device.strip():
            raise ValueError(tr("Select a connected scanner first."))
        if self.driver not in {"twain", "wia", "sane", "apple"}:
            raise ValueError(tr("Choose a supported scanner connection."))
        if self.dpi not in {150, 300, 600, 1200} or not 1 <= self.scale <= 100:
            raise ValueError(tr("Choose 150–1200 DPI and an output scale between 1% and 100%."))
        if self.driver == "wia" and self.dpi > 600:
            raise ValueError(
                tr("The LiDE 300 WIA driver supports up to 600 DPI. Choose TWAIN for 1200 DPI.")
            )
        if self.page not in {"216x297mm", "a4", "letter"} or self.mode not in {"color", "gray"}:
            raise ValueError(tr("Choose a supported page size and color mode."))


def scanner_command():
    override = os.environ.get("PORTRAIT_SCANNER_CONSOLE")
    if override and Path(override).is_file():
        return [override]
    if sys.platform == "win32":
        for path in (
            Path(sys.executable).parent.parent / "scanner" / "App" / "NAPS2.Console.exe",
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "NAPS2"
            / "NAPS2.Console.exe",
        ):
            if path.is_file():
                return [str(path)]
    else:
        executable = shutil.which("naps2")
        if executable:
            return [executable, "console"]
    raise RuntimeError(
        tr(
            "Scanner support was not found. Use the latest Windows portable build, or install NAPS2 for a source/Linux installation."
        )
    )


PROFILE_NAME = "PortraitProcessor"

# NAPS2's built-in defaults (--noprofile) keep each scanned page as a quality-75 JPEG
# before writing the output, so even a TIFF/PNG sheet carried JPEG artifacts. This
# profile keeps pages lossless; the command-line options override everything else.
PROFILES_XML = f"""<?xml version="1.0"?>
<ArrayOfScanProfile xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <ScanProfile>
    <Version>2</Version>
    <DisplayName>{PROFILE_NAME}</DisplayName>
    <IsDefault>true</IsDefault>
    <MaxQuality>true</MaxQuality>
    <Quality>100</Quality>
    <UseNativeUI>false</UseNativeUI>
    <PaperSource>Glass</PaperSource>
  </ScanProfile>
</ArrayOfScanProfile>
"""


def scanner_data_directory():
    return model_directory().parent / "naps2"


class ScannerBackend:
    def __init__(self, command=None, data_directory=None):
        self.command = command or scanner_command()
        self.data_directory = Path(data_directory or scanner_data_directory())

    def _environment(self):
        """Give NAPS2 a private data folder so its profile never touches the user's NAPS2."""
        self.data_directory.mkdir(parents=True, exist_ok=True)
        (self.data_directory / "profiles.xml").write_text(PROFILES_XML, encoding="utf-8")
        return {**os.environ, "NAPS2_TEST_DATA": str(self.data_directory)}

    @staticmethod
    def _stop(process):
        if process.poll() is not None:
            return
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=5,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def _run(self, arguments, cancel, timeout):
        check_cancel(cancel)
        process = subprocess.Popen(
            [*self.command, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._environment(),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )
        started = time.monotonic()
        try:
            while True:
                check_cancel(cancel)
                if time.monotonic() - started > timeout:
                    raise TimeoutError(
                        tr(
                            "The scanner did not respond in time. Check its USB connection and try again."
                        )
                    )
                try:
                    stdout, stderr = process.communicate(timeout=0.15)
                    break
                except subprocess.TimeoutExpired:
                    continue
            text = stdout.decode("utf-8-sig", errors="replace").strip()
            errors = stderr.decode("utf-8-sig", errors="replace").strip()
            if process.returncode:
                raise RuntimeError((text or errors or tr("The scanner operation failed."))[-1200:])
            return text
        finally:
            self._stop(process)
            process.stdout.close()
            process.stderr.close()

    def devices(self, driver, cancel):
        text = self._run(["--listdevices", "--driver", driver], cancel, 45)
        if re.search(
            r"unexpected error|could not load|unable to|failed to|no devices found",
            text,
            re.IGNORECASE,
        ):
            raise RuntimeError(text[-1200:])
        return list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))

    def capture(self, settings, path, cancel, *, preview=False):
        if preview:
            settings = replace(settings, dpi=150, scale=100)
        settings.validate()
        path = Path(path)
        text = self._run(
            [
                "--profile",
                PROFILE_NAME,
                "--driver",
                settings.driver,
                "--device",
                settings.device,
                "--source",
                "glass",
                "--dpi",
                str(settings.acquisition_dpi),
                "--pagesize",
                settings.page,
                "--bitdepth",
                settings.mode,
                "--tiffcomp",
                "none",
                "--disableocr",
                "--number",
                "1",
                "--output",
                str(path),
            ],
            cancel,
            900,
        )
        check_cancel(cancel)
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(
                text[-1200:]
                or tr("No image was received. Check the Canon driver and USB connection.")
            )
        with Image.open(path) as image:
            if image.width < 8 or image.height < 8:
                raise RuntimeError(tr("The scanner returned an empty image."))
        return path


def scan_preview(path):
    with Image.open(path) as source:
        source.thumbnail((1400, 1400))
        return np.array(source.convert("RGB"))


def save_capture(source, directory, name, settings, image_format, cancel):
    settings.validate()
    stem = export_stem(name)
    extensions = {"PNG": "png", "TIFF": "tiff", "JPEG": "jpg"}
    if image_format not in extensions:
        raise ValueError(tr("Choose PNG, TIFF, or JPEG for the scanned sheet."))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=".scan-", suffix="." + extensions[image_format], dir=directory
    )
    os.close(handle)
    temporary = Path(temporary)
    try:
        with Image.open(source) as original:
            image = original if original.mode == "RGB" else original.convert("RGB")
            factor = settings.output_dpi / settings.acquisition_dpi
            if factor < 1:
                size = tuple(max(1, round(n * factor)) for n in image.size)
                image = image.resize(size, Image.Resampling.LANCZOS)
            check_cancel(cancel)
            options = {"dpi": (settings.output_dpi,) * 2}
            if image_format == "PNG":
                options["compress_level"] = 1
            elif image_format == "JPEG":
                options.update(quality=95, subsampling=0, optimize=False)
            else:
                options["compression"] = "tiff_lzw"
            image.save(temporary, format=image_format, **options)
        check_cancel(cancel)
        index = 1
        while True:
            suffix = "" if index == 1 else f"_{index}"
            destination = directory / f"{stem}{suffix}.{extensions[image_format]}"
            try:
                if sys.platform == "win32":
                    os.rename(temporary, destination)
                else:
                    os.link(temporary, destination)
                return destination
            except FileExistsError:
                index += 1
    finally:
        temporary.unlink(missing_ok=True)
