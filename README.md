# Portrait Processor

A local Windows and Linux desktop editor for human portraits. Python handles the UI and AI;
a Rust extension handles canvas placement, rotation, resampling, and optional
symmetric foreground trimming.

## Launch

### Windows portable app

Extract `dist/PortraitProcessor-Windows-x64.zip`, then double-click
**PortraitProcessor.exe** inside the extracted folder. Keep its `runtime` folder
beside the executable. Keep the `scanner` folder there as well. Python, Rust,
Qt, the AI libraries, and the NAPS2 scanner bridge are bundled.
The first AI operation downloads model weights; later runs reuse the cache.

This x64 package targets Windows 10 and 11. It has been checked under Wine,
including real background removal, face detection, manual positioning, and exports;
a native Windows machine has not been used for verification.
Windows logs are in `%LOCALAPPDATA%\PortraitProcessor\logs\app.log`.

### Linux / source checkout

```sh
cd /path/to/portrait-processor
./setup.sh   # first installation, or after updating dependencies
./run.sh
```

The environment uses Python 3.12 through `uv`. Rust and `uv` must be installed.
Dependencies are locked in `uv.lock` and `Cargo.lock`; the Rust extension builds
as part of setup. Ordinary launches use the existing environment without network
access or dependency resolution.
Setup also adds **Portrait Processor** to your Linux application menu.

Optionally open specific files and choose the output directory:

```sh
./run.sh /path/to/portrait.jpg --output /path/to/processed_photos
./run.sh /path/to/photo_folder
./run.sh --scan /path/to/scanned_sheet.jpg --output /path/to/processed_photos
```

## Workflow

For a sheet containing several printed portraits, choose **Import scan…**.
The whole sheet is rotated 180° first. Photo borders are detected at several
detail levels, then shown in a separate review screen. Click an outline and
drag its corners or move the whole crop. **Add missing photo** lets you draw a
rectangle; **Remove selected** discards a false detection. Scroll to zoom and
use **Fit sheet** to see the whole scan. If the source was already upright,
**Rotate sheet 180°** switches it back before import.

**Import portraits** separates and straightens complete prints at their scanned
resolution and numbers them in reading order (top to bottom, left to right).
It prepares the first five portraits, one at a time, then unlocks the editor
and automatically prepares subsequent groups in the background. Moving,
renaming, choosing formats, and saving prepared portraits remain available.
If you reach an unfinished portrait, only that selection waits; it unlocks as
soon as its result is ready. You can return to any prepared portrait meanwhile.
Nothing is saved automatically. **Pause preparation** stops background work and
keeps completed results and edits; **Resume preparation** continues it.
Pause preparation before changing accuracy, retrying AI, or opening another queue.
One AI worker runs at a time, with rendering and file loading on a separate queue.
The AI process exits between groups to release its allocations.
Detection uses paper borders rather than brightness or skin colors. Faint,
overlapping, or damaged borders may need manual correction. Check the crop
preview includes all hair and clothing. Portraits do not need to form a fixed grid.

After importing, review each portrait in the normal editor, enter its name,
choose a format, and **Save & next**. The original scan stays untouched.
Individual source crops and edit drafts are temporary; save your exports before
closing. The output folder defaults to `processed_photos` beside the original
scan, rather than inside the temporary crop folder. Photos needing manual face
markers are flagged in the queue. **Process next 5** also prepares an ordinary
photo queue and reuses masks already processed. Cancellation retains completed
portraits; run it again to prepare the rest.

In **Final portrait** or **Before trimming**, drag the portrait directly inside
its 948×1048 canvas. **Snap** optionally aligns it to the center and frame edges;
hold Alt to bypass snapping for a drag. Hold Space to pan the view instead, and
scroll to zoom the view. **Left / right**, **Up / down**, and **Reset position**
under Retouch & alignment provide precise placement controls. Translation keeps
the existing size and mask, without repeating AI or resampling the pixels.
Move a portrait down slightly to cover a white wedge below a tilted lower edge.
Position is retained per photo, including when the photo is evicted from memory.

1. Open photos or a folder. Supported formats: JPEG, PNG, WebP, TIFF, BMP.
   Folder opening includes files directly inside that folder.
2. The app downloads its models on first use, removes the background locally,
   and detects the nose and eyes. The first inference can take noticeably longer
   while dependencies and models initialize. CPU speed depends on the image and
   hardware; model inference is the main cost.
3. Review **Final portrait** and **Before trimming**. The portrait builds upward
   from the bottom of the canvas. Adjust **Head top limit** (default 180 px below
   the top), **Side inset** (64 px on each side), and **Bottom gap** (0 px).
   The head stays below the horizontal guide and the portrait stays between the
   vertical guides. Use **Fine rotation** under **Retouch & alignment** to correct
   tilt. A photo keeps its original scale unless
   shrinking is necessary to fit the foreground and its margins. It is never
   automatically enlarged. The canvas is always 948×1048.
4. Expand **Retouch & alignment** to correct mistakes in **Original & mask**. Use **Nose**, **Eye 1**, and **Eye 2** to
   place markers; click either eye first. **Erase** removes unwanted foreground;
   **Restore** brings back original image pixels, including hair mistakenly
   removed by AI. Pink tint indicates excluded background. Brush diameter and
   particle area are measured in original-source pixels. Undo/redo applies to
   brush strokes and particle cleanup. **Left / right** shifts the portrait for
   visual balance; positive values move it right. Positive **Up / down** values
   move it down. Manual placement can go beyond the guides and crop at the canvas
   edge. It does not repeat AI inference. **Manual setup** remains available
   after AI processing and keeps the current mask.
5. Enter a **Save as** name (the original filename is the default), then click
   **Save** or **Save & next** after reviewing. Choose **PNG**, **JPG/JPEG**,
   **WebP**, **TIFF**, or **BMP**. Typing a known extension also selects that format.
   JPEG and WebP have a quality control; PNG, TIFF, and BMP are lossless.
   Names and export settings are remembered per photo. Exports have a
   white background, exactly 948×1048 pixels, without the editor guide or markers.
   The default destination is `processed_photos` beside the input images.

The automatic rotation levels the eyes. It corrects tilt, not a head turned
sideways. If no face is found, set all three markers manually. For several faces,
choose one in the face selector and erase any unwanted people from the mask.
If AI cannot run, **Manual setup** enables manual masking and marker placement.
Reset returns to the initial AI mask, detected landmarks, and default spacing.

**Accuracy** offers five background-removal levels, from fastest with rougher
edges to slowest with the most detailed edges: **Fastest** (U²-Net human
segmentation), **Fast · refined edges** (the same model with edge refinement),
**Balanced** (IS-Net), **Detailed** (BiRefNet Lite), and **Maximum** (BiRefNet
Portrait). AI uses up to two CPU threads. In **Auto**, enter the time per photo
you accept; **Measure this PC** runs a quick timing in the background and Auto
picks the most detailed level that fits, showing the predicted time. The
measurement is stored per PC, so changing the budget re-picks instantly; moving
the settings to another computer measures again. Auto only picks levels 1–4.
Inspect difficult hair edges and choose a higher level or Restore if needed.
Changing the level applies to future processing and keeps existing masks and edits.
The first photo also loads model weights; a slower
PC may take several minutes per portrait. The status shows the current stage,
the portrait number during a batch, and elapsed time. AI runs in a separate,
reusable process so Qt remains responsive; **Cancel** stops that process and
keeps completed batch results. A retry starts a fresh worker. An operation
exceeding 15 minutes returns control to the editor for retry or manual setup.
Unused audio-device initialization from the face library is disabled, avoiding
an unrelated stall when loading facial landmarks. Inference uses CPU memory
settings that release activation allocations between portraits.

## Direct scanning

Choose **Scanners…**, connect the CanoScan LiDE 300 by USB, and select the
device. Install Canon's scanner driver first; IJ Scan Utility can stay closed.
Windows uses the bundled NAPS2 command-line bridge and **TWAIN** by default,
including its 32-bit driver worker.

Set resolution (150, 300, 600, or 1200 DPI), output scale (default **100%**),
color, and scan area. Full bed is 216×297 mm. **49%** makes each dimension 49%
of the full-resolution pixels; it is separate from JPEG compression. Like Canon
ScanGear's output size, a reduced scale reads the glass at the lowest sufficient
resolution: 1200 DPI at 49% (588 DPI effective) scans at 600 DPI and resizes the
result slightly, which is several times faster than a true 1200 DPI pass.
**Preview** uses 150 DPI. **Scan & import portraits** captures at the selected
settings, saves the sheet in the chosen folder as PNG, TIFF, or JPEG, and opens
crop review followed by the five-portrait workflow. Individual portraits retain
their own names and formats. Cancelling stops acquisition; saved sheets remain.

Canon documents a 50–600 DPI range with WIA for this model. The WIA connection
therefore caps choices at 600 DPI; choose TWAIN for 1200 DPI. Actual acquisition
depends on the installed driver and connected scanner. Simulated acquisition
and the bundled command-line bridge are tested; a physical LiDE 300 has not
been available here.

Source/Linux installations need NAPS2 installed separately; Linux uses SANE.
`PORTRAIT_SCANNER_CONSOLE` can specify a custom console executable.
The unmodified Windows NAPS2 binaries, licenses, notices, and corresponding
source archive are included under `scanner`. Run `tools/fetch_scanner_runtime.py`
before building scanner support; its download is pinned and checksum verified.

References: [Canon's WIA limits](https://ij.manual.canon/ij/webmanual/ScanUtility/W/LiDE%20300/1.1/EN/SU/su-401.html),
[NAPS2 scanner controls](https://www.naps2.com/doc/command-line).

## Hair preservation and optional symmetry

The nose is aligned to the geometric center: between columns 473 and 474 when
counting from zero, plus any manual **Left / right** adjustment. A blue preview
line marks the adjusted nose position. By default the complete background-removal mask is retained,
including asymmetric hair, ears, and shoulders. The top and side limits still apply.

**Strict outline symmetry** under **Retouch & alignment** is optional. Enabling
it can remove hair, because real hairstyles have different outlines on each side.
For each pixel and its horizontally reflected partner, the
final foreground opacity is the smaller of the two original opacities:

```text
final_alpha[y, x] = min(alpha[y, x], alpha[y, 947 - x])
```

This only removes foreground. It does not mirror facial colors or change the
canvas width. It can trim asymmetric hair, ears, or shoulders, and remove details
beside unmatched holes in the opposite half. Inspect **Before trimming**
before saving; manual source-mask restoration can improve the result.

Geometry uses the foreground mask, not brightness. A white shirt is still part
of the subject. Soft edges are resampled using premultiplied color to avoid
mixing hidden background pixels into rotation edges. The top and side guides are
limits, so a smaller portrait may leave more space. The solid lower foreground
edge is aligned to the bottom gap, including after rotation. The faint fringe
at a photographed or rotated bottom edge is cropped at the canvas boundary,
preventing a thin white strip underneath the subject. This uses mask opacity,
so light clothing is preserved.
All frame guides are preview overlays and are excluded from saved photos.

## Saving and edit state

The app never changes original files. Existing exports receive `_2`, `_3`, etc.
Output is written completely before it is published under its final name.
The current preview must finish before Save is enabled.

Edits remain available while switching among photos in the current queue. They
are kept in a bounded memory cache and temporary session drafts; closing or replacing
a queue prompts about unsaved work. Temporary drafts are deleted when the app closes.
Source editing sessions are not persisted after closing the app. To revise a
photo later, reopen its original. A queued photo's check mark indicates that an
export exists; Save remains available for additional exports.

## Models and offline use

- Background: [rembg](https://github.com/danielgatis/rembg) models
  `u2net_human_seg`, `isnet-general-use`, `birefnet-general-lite`, and
  `birefnet-portrait`.
- Landmarks: [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker),
  using Google's versioned float16 model asset.

Model weights are cached. Face landmarks are stored beneath
`${XDG_CACHE_HOME:-~/.cache}/portrait-processor/models`; rembg manages its own cache.
On Windows, both models use `%LOCALAPPDATA%\PortraitProcessor\Cache\models`.
The Windows package bundles levels 1–4 as half-precision files under `models`;
each is expanded once into the cache on first use, so they work offline.
Level 5 (Maximum) downloads about 970 MB on first use.
Photos are processed on this computer, not submitted to a background-removal API.
The libraries and model weights retain their upstream licenses.

Download the models ahead of time:

```sh
.venv/bin/python tools/download_models.py
```

If a download fails, check connectivity and press **Retry AI**. Manual
editing remains available. Cancellation waits for the current native model
operation to return, then discards its result.

## Development and checks

```sh
uv sync --locked --dev
cargo test
cargo clippy --all-targets -- -D warnings
QT_QPA_PLATFORM=offscreen .venv/bin/pytest
.venv/bin/ruff check python tests tools
```

Rebuild the native extension after changing Rust:

```sh
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --release --uv
```

Run real model inference on scikit-image's bundled public-domain astronaut photo:

```sh
.venv/bin/python tools/verify_models.py
```

It writes source, mask, before-trim, and processed examples into `artifacts/`.
Unit and desktop tests do not require downloaded models.
For a real desktop check, `tools/desktop_smoke.py` opens the editor, processes the
same sample through its worker, captures the UI, saves a sample, and closes.
MediaPipe initializes native graphics resources even with CPU inference, so run
these real-model checks in a normal desktop session rather than a restrictive sandbox.

### Build the portable Windows package on Linux

Install MinGW-w64 (including GCC and windres), uv, and the Rust Windows target:

```sh
rustup target add x86_64-pc-windows-gnu
uv run python tools/pack_models.py
.venv/bin/python tools/build_windows.py
```

`tools/pack_models.py` writes the half-precision bundled models to
`artifacts/models`; the build stops if any are missing.

The builder downloads a private CPython 3.12.15 Windows runtime through uv,
installs the hashed Windows dependency lock, cross-compiles the Rust extension,
embeds the launcher icon, and creates the ZIP and SHA-256 file in `dist/`.
It uses PyO3's [Windows import-library support](https://pyo3.rs/v0.27.2/building-and-distribution.html)
and uv's [managed Python distributions](https://docs.astral.sh/uv/concepts/python-versions/).
Rebuilding does not publish source code or artifacts to an external service.

### Build the portable Linux or macOS package

On the target system (Linux x64, or macOS 14+ on Apple silicon), after packing the models:

```sh
uv run --no-sync python tools/build_portable.py --self-test
```

Linux gets `dist/PortraitProcessor-Linux-x64.tar.gz`: extract it, run `./PortraitProcessor`,
and optionally `./install-shortcut.sh` to add it to the application menu. It needs glibc 2.34+
(Ubuntu 22.04, Debian 12, Fedora 35 or newer). macOS gets `dist/PortraitProcessor-macOS-arm64.zip`
holding an unsigned `Portrait Processor.app`; on first launch, right-click it and choose Open,
or run `xattr -dr com.apple.quarantine "Portrait Processor.app"`.

### Releases

`.github/workflows/release.yml` builds and tests all three packages on GitHub Actions.
Push a version tag to publish a GitHub release with the packages and their SHA-256 files:

```sh
git tag v0.1.0
git push origin v0.1.0
```

Running the workflow by hand from the Actions tab builds and tests the packages without
publishing; download them from the run's Artifacts section.
