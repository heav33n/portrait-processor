PORTRAIT PROCESSOR — WINDOWS x64

1. Extract the entire ZIP to a folder.
2. Open PortraitProcessor.exe.
3. Add photos or open a folder, review the result, choose a filename and format,
   then click Save photo or Save & next.

For a scanned sheet, choose Import scan. The sheet rotates 180 degrees, then
shows numbered photo outlines. Adjust corners, move crops, or add/remove a
boundary. Import portraits prepares the first five, then unlocks editing while
the next groups prepare in the background, one portrait at a time. Nothing is
saved automatically. Name, choose formats, move, and save prepared portraits.
An unfinished selection waits only until its result is ready. Pause preparation
keeps completed results; Resume preparation continues. Pause first to change AI
mode, retry AI, or open another queue. Drag portraits in the final canvas.
Snap aligns to the center and frame edges; hold Alt to bypass, or switch Snap off.
Hold Space to pan the view. Retouch & alignment also has Left / right, Up / down,
and Reset position. Moving does not change size or repeat AI.
Individual crops are temporary; save your exports before closing the app.
The scan is never modified. If it was already upright, use Rotate sheet 180
degrees in the crop review screen before importing.

Python, Rust, Qt, and the AI libraries are included. No installation or
administrator access is required. Keep the runtime and scanner folders next to
the EXE. Canon's scanner driver is required for direct scanner access.

The first AI operation downloads model weights (about 176 MB for Lower memory,
or about 1 GB for Detailed edges). Subsequent use
reuses those files. Your photos are processed locally.

Detailed background removal runs on the CPU and may take several minutes on
slower PCs. Status includes elapsed time and the current stage. Cancel stops
the separate AI process; completed batch results are retained for review or
resuming. Retry AI starts it again. Manual setup remains available afterward.

Export formats: PNG, JPG/JPEG, WebP, TIFF, BMP. JPEG and WebP offer quality controls.
Output size: 948x1048. Background: white. Natural hair is preserved by default.

Lower memory is the default mode and limits CPU use to two threads. Detailed
edges is available for difficult hair masks. Use Restore for manual corrections.
The solid bottom edge is automatically cropped below a tilted scan border to
remove the white wedge. Manual Up / down placement remains available.

Choose Scanners for direct LiDE 300 scanning. Connect USB and install the Canon
driver. TWAIN supports the 1200 DPI option; WIA on this scanner is capped at 600.
Output scale defaults to 100%; 49% is also supported. Preview uses 150 DPI.
Scan & import portraits saves the sheet first, then opens crop review. You can
set its name, PNG/TIFF/JPEG format, and folder. Portrait exports stay separate.
The NAPS2 bridge and its corresponding source/license are under scanner.

For uneven hairstyles, open Retouch & alignment and adjust Left / right.
Positive values move the portrait right. Manual setup also works after AI runs.
Strict outline symmetry is optional and can remove asymmetric hair.

Default exports go into processed_photos beside the source images.
Existing photos are never overwritten; numbered suffixes are added.

Logs: %LOCALAPPDATA%\PortraitProcessor\logs\app.log
Model cache: %LOCALAPPDATA%\PortraitProcessor\Cache\models

Third-party software and runtime licenses are included with their respective
packages under runtime. This package contains no personal source photos or AI
model weights.
