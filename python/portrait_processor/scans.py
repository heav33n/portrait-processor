"""Split scanned sheets without using skin, hair, or background color as a crop."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .ai import check_cancel
from .i18n import tr


@dataclass
class Scan:
    path: Path
    rgb: np.ndarray
    crops: list[np.ndarray]  # Four clockwise source-pixel corners, starting at top left.


def ordered_corners(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0).copy()


def reading_order(crops):
    """Group slightly tilted prints into rows, then read each row left to right."""
    rows = []
    for crop in sorted(crops, key=lambda c: float(c[:, 1].mean())):
        height = float(np.ptp(crop[:, 1]))
        cy = float(crop[:, 1].mean())
        row = next((r for r in rows if abs(cy - r[0]) < min(height, r[1]) * 0.45), None)
        if row is None:
            rows.append([cy, height, [crop]])
        else:
            row[2].append(crop)
    return [c for _, _, row in rows for c in sorted(row, key=lambda c: float(c[:, 0].mean()))]


def detect_crops(rgb, progress=lambda message: None, cancel=None):
    """Find paper edges at several scales, including faint white-on-white borders."""
    import cv2

    cancel = cancel if cancel is not None else threading.Event()
    height, width = rgb.shape[:2]
    crops = []
    # A faint paper boundary may close at one scale but merge with dust at another.
    for size in sorted({min(max(height, width), n) for n in (1600, 2400, 3200)}):
        check_cancel(cancel)
        progress(tr("Finding individual photo borders · detail {size} px…", size=size))
        scale = size / max(height, width)
        small = cv2.resize(rgb, (round(width * scale), round(height * scale)))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (3, 3), 0)
        area = gray.size
        for low, high in ((3, 12), (5, 18), (8, 25), (12, 35), (20, 60)):
            edges = cv2.Canny(gray, low, high)
            for kernel in (3, 7, 11):
                check_cancel(cancel)
                closed = cv2.morphologyEx(
                    edges, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8)
                )
                contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_area = cv2.contourArea(contour)
                    if not area * 0.006 < contour_area < area * 0.45:
                        continue
                    _, (w, h), _ = cv2.minAreaRect(contour)
                    if min(w, h) < 40 or min(w, h) / max(w, h) < 0.55:
                        continue
                    if contour_area / (w * h) < 0.87:
                        continue
                    approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)
                    if len(approx) != 4 or not cv2.isContourConvex(approx):
                        continue
                    crop = ordered_corners(approx) / scale
                    top_width = np.linalg.norm(crop[1] - crop[0])
                    left_height = np.linalg.norm(crop[3] - crop[0])
                    if not 0.5 < top_width / left_height < 1.3:
                        continue
                    # Suppress repeated inner/outer paper borders and nested contours.
                    if any(
                        cv2.intersectConvexConvex(crop, old)[0]
                        / min(cv2.contourArea(crop), cv2.contourArea(old))
                        > 0.65
                        for old in crops
                    ):
                        continue
                    crop[:, 0] = np.clip(crop[:, 0], 0, width - 1)
                    crop[:, 1] = np.clip(crop[:, 1], 0, height - 1)
                    crops.append(crop)
    return reading_order(crops)


def load_scan(path, progress=lambda message: None, cancel=None):
    cancel = cancel if cancel is not None else threading.Event()
    progress(tr("Opening scan and rotating the whole sheet 180°…"))
    with Image.open(path) as source:
        ImageOps.exif_transpose(source, in_place=True)
        if source.mode == "RGB":
            rgb = np.asarray(source)
        else:
            with source.convert("RGB") as color:
                rgb = np.asarray(color)
    # Close the decoder before rotating; full-resolution 1200 DPI sheets are large.
    rgb = np.ascontiguousarray(rgb[::-1, ::-1])
    check_cancel(cancel)
    return Scan(Path(path).resolve(), rgb, detect_crops(rgb, progress, cancel))


def extract_crop(rgb, corners):
    """Straighten a complete print at its scanned resolution; never crop to a face."""
    import cv2

    points = ordered_corners(corners)
    if not cv2.isContourConvex(points) or cv2.contourArea(points) < 64:
        raise ValueError(tr("A crop must have four uncrossed corners and a nonzero area."))
    w = round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
    h = round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
    if min(w, h) < 8:
        raise ValueError(tr("This crop is too small. Enlarge its boundary or remove it."))
    target = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(
        rgb,
        transform,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def extract_scan(scan, crops, directory, progress=lambda message: None, cancel=None):
    cancel = cancel if cancel is not None else threading.Event()
    paths = []
    for index, corners in enumerate(reading_order(crops), 1):
        check_cancel(cancel)
        progress(tr("Separating portrait {index} of {count}…", index=index, count=len(crops)))
        crop = extract_crop(scan.rgb, corners)
        path = Path(directory) / f"{scan.path.stem}_{index:02d}.png"
        Image.fromarray(crop).save(path)
        paths.append(path)
    check_cancel(cancel)
    return paths
