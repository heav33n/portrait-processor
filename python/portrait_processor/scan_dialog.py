"""Review and adjust complete print boundaries before running portrait AI."""

from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .canvas import pixmap
from .i18n import tr
from .scans import extract_crop, ordered_corners, reading_order


class ScanCanvas(QGraphicsView):
    changed = Signal()

    def __init__(self, scan, parent=None):
        super().__init__(parent)
        self.scan = scan
        self.crops = [c.copy() for c in scan.crops]
        self.selected = 0 if self.crops else -1
        self.adding = False
        self.drag = None
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#101714"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(350, 320)
        self.show_scan()

    def show_scan(self):
        import cv2

        self.scene().clear()
        h, w = self.scan.rgb.shape[:2]
        scale = min(1, 1800 / max(h, w))
        preview = cv2.resize(self.scan.rgb, (round(w * scale), round(h * scale)))
        item = self.scene().addPixmap(pixmap(preview))
        item.setScale(w / preview.shape[1])
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        self.fit()

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if 0.025 <= self.transform().m11() * factor <= 5:
            self.scale(factor, factor)
        event.accept()

    def drawForeground(self, painter, rect):
        scale = max(self.transform().m11(), 0.001)
        for index, crop in enumerate(self.crops):
            color = QColor("#a4edbc" if index == self.selected else "#ff9386")
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(
                QColor(20, 50, 30, 28) if index == self.selected else Qt.BrushStyle.NoBrush
            )
            painter.drawPolygon(QPolygonF([QPointF(*p) for p in crop]))
            painter.save()
            painter.translate(float(crop[:, 0].min()), float(crop[:, 1].min()))
            painter.scale(1 / scale, 1 / scale)
            painter.fillRect(QRectF(0, 0, 35, 23), QColor("#172b1e"))
            painter.drawText(QRectF(0, 0, 35, 23), Qt.AlignmentFlag.AlignCenter, str(index + 1))
            painter.restore()
            if index == self.selected:
                painter.setBrush(QColor("#172b1e"))
                for point in crop:
                    painter.drawEllipse(QPointF(*point), 5 / scale, 5 / scale)

    def point(self, event):
        p = self.mapToScene(event.position().toPoint())
        h, w = self.scan.rgb.shape[:2]
        return np.array([np.clip(p.x(), 0, w - 1), np.clip(p.y(), 0, h - 1)], dtype=np.float32)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        point = self.point(event)
        if self.adding:
            self.drag = ("new", point)
            self.crops.append(np.tile(point, (4, 1)))
            self.selected = len(self.crops) - 1
        else:
            if self.selected >= 0:
                distances = np.linalg.norm(self.crops[self.selected] - point, axis=1)
                corner = int(np.argmin(distances))
                if distances[corner] * self.transform().m11() < 14:
                    self.drag = ("corner", corner)
                    return
            self.selected = next(
                (
                    i
                    for i in reversed(range(len(self.crops)))
                    if QPolygonF([QPointF(*p) for p in self.crops[i]]).containsPoint(
                        QPointF(*point), Qt.FillRule.OddEvenFill
                    )
                ),
                -1,
            )
            if self.selected >= 0:
                self.drag = ("move", point, self.crops[self.selected].copy())
        self.changed.emit()
        self.viewport().update()

    def mouseMoveEvent(self, event):
        if self.drag is None:
            return super().mouseMoveEvent(event)
        point = self.point(event)
        kind, start, *original = self.drag
        if kind == "corner":
            candidate = self.crops[self.selected].copy()
            candidate[start] = point
            import cv2

            if cv2.isContourConvex(candidate) and cv2.contourArea(candidate) >= 64:
                self.crops[self.selected] = candidate
        elif kind == "new":
            x1, y1 = np.minimum(start, point)
            x2, y2 = np.maximum(start, point)
            self.crops[self.selected] = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )
        else:
            offset = point - start
            crop = original[0]
            h, w = self.scan.rgb.shape[:2]
            offset = np.clip(offset, -crop.min(axis=0), [w - 1, h - 1] - crop.max(axis=0))
            self.crops[self.selected] = crop + offset
        self.changed.emit()
        self.viewport().update()

    def mouseReleaseEvent(self, event):
        if self.drag is not None:
            self.drag = None
            self.adding = False
            if self.selected >= 0 and min(np.ptp(self.crops[self.selected], axis=0)) < 8:
                self.remove_selected()
            if self.selected >= 0:
                selected = self.crops[self.selected]
                self.crops = reading_order(self.crops)
                self.selected = next(i for i, crop in enumerate(self.crops) if crop is selected)
            self.changed.emit()
            self.viewport().update()
        super().mouseReleaseEvent(event)

    def remove_selected(self):
        if self.selected >= 0:
            self.crops.pop(self.selected)
            self.selected = min(self.selected, len(self.crops) - 1)
            self.changed.emit()
            self.viewport().update()

    def rotate_half_turn(self):
        h, w = self.scan.rgb.shape[:2]
        self.scan.rgb = np.ascontiguousarray(self.scan.rgb[::-1, ::-1])
        self.crops = reading_order([ordered_corners([w - 1, h - 1] - c) for c in self.crops])
        self.selected = 0 if self.crops else -1
        self.show_scan()
        self.changed.emit()


class ScanReview(QDialog):
    def __init__(self, scan, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Separate scanned portraits"))
        self.resize(1120, 840)
        self.setMinimumSize(780, 600)
        body = QVBoxLayout(self)
        body.setContentsMargins(24, 22, 24, 22)
        body.setSpacing(16)
        title = QLabel(tr("Review your scanned sheet"))
        title.setObjectName("title")
        body.addWidget(title)
        description = QLabel(
            tr("Sheet rotated 180° · Each outline becomes one portrait. Originals stay unchanged.")
        )
        description.setObjectName("muted")
        description.setWordWrap(True)
        body.addWidget(description)
        row = QHBoxLayout()
        row.setSpacing(20)
        self.canvas = ScanCanvas(scan)
        row.addWidget(self.canvas, 1)
        sidebar = QWidget()
        sidebar.setFixedWidth(225)
        controls = QVBoxLayout(sidebar)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(12)
        self.count = QLabel()
        self.count.setObjectName("cardTitle")
        controls.addWidget(self.count)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(240)
        controls.addWidget(self.preview)
        text = QLabel(
            tr(
                "Click an outline to select it.\nDrag its corners to adjust the crop.\nDrag inside to move it. Scroll to zoom.\n\nKeep all hair and clothing inside."
            )
        )
        text.setObjectName("muted")
        text.setWordWrap(True)
        controls.addWidget(text)
        self.add_button = QPushButton(tr("+ Add missing photo"))
        self.add_button.setCheckable(True)
        self.add_button.toggled.connect(self._add_mode)
        controls.addWidget(self.add_button)
        self.remove_button = QPushButton(tr("Remove selected"))
        self.remove_button.clicked.connect(self.canvas.remove_selected)
        controls.addWidget(self.remove_button)
        rotate = QPushButton(tr("Rotate sheet 180°"))
        rotate.clicked.connect(self.canvas.rotate_half_turn)
        controls.addWidget(rotate)
        fit = QPushButton(tr("Fit sheet"))
        fit.clicked.connect(self.canvas.fit)
        controls.addWidget(fit)
        controls.addStretch()
        row.addWidget(sidebar)
        body.addLayout(row, 1)
        footer = QHBoxLayout()
        hint = QLabel(tr("Prepare five portraits at a time. Review, name, and save each yourself."))
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        footer.addWidget(hint, 1)
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.reject)
        footer.addWidget(cancel)
        self.import_button = QPushButton()
        self.import_button.setObjectName("primary")
        self.import_button.clicked.connect(self.accept)
        footer.addWidget(self.import_button)
        body.addLayout(footer)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self._preview)
        self.canvas.changed.connect(self._changed)
        self._changed()

    def _add_mode(self, enabled):
        self.canvas.adding = enabled
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _changed(self):
        count = len(self.canvas.crops)
        self.count.setText(
            tr("1 portrait detected")
            if count == 1
            else tr("{count} portraits detected", count=count)
        )
        self.import_button.setText(tr("Import {count} · prepare first 5", count=count))
        self.import_button.setEnabled(count > 0 and self.canvas.drag is None)
        self.remove_button.setEnabled(self.canvas.selected >= 0)
        if not self.canvas.adding:
            self.add_button.setChecked(False)
        self.timer.start()

    def _preview(self):
        if self.canvas.selected < 0:
            self.preview.clear()
            self.preview.setText(
                tr("Add a photo boundary\nto begin.")
                if not self.canvas.crops
                else tr("Select a portrait")
            )
            return
        try:
            crop = extract_crop(self.canvas.scan.rgb, self.canvas.crops[self.canvas.selected])
            image = Image.fromarray(crop)
            image.thumbnail((225, 260))
            self.preview.setPixmap(pixmap(np.asarray(image)))
        except ValueError:
            self.preview.setText(tr("Adjust this crop's corners."))
