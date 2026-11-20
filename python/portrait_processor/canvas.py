from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView

from .document import translate_layer
from .i18n import tr


def pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class Canvas(QGraphicsView):
    point_selected = Signal(float, float)
    stroke_started = Signal()
    brush_moved = Signal(float, float)
    stroke_finished = Signal()
    portrait_drag_started = Signal()
    portrait_moved = Signal(float, float, bool)
    portrait_drag_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QColor("#101714"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMinimumSize(320, 260)
        self.tool = "pan"
        self.painting = False
        self.auto_fit = True
        self.image_item = None
        self.markers = []
        self.guide = None
        self.frame_guides = []
        self.size = None
        self.portrait_drag = None
        self.space_tool = None
        self.rgb = None
        self.empty()

    def empty(self):
        self.scene().clear()
        self.image_item = None
        self.markers = []
        self.guide = None
        self.frame_guides = []
        self.size = None
        text = self.scene().addText(
            tr("Open a portrait to begin\n\nChoose photos or a folder above.")
        )
        text.setDefaultTextColor(QColor("#9aafa3"))
        self.scene().setSceneRect(text.boundingRect().adjusted(-70, -120, 70, 120))
        self.fit()

    def set_tool(self, tool):
        self.tool = tool
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.setCursor(
            Qt.CursorShape.OpenHandCursor
            if tool in {"pan", "portrait"}
            else Qt.CursorShape.CrossCursor
        )

    def show_image(
        self,
        rgb,
        markers=None,
        guide=False,
        preserve_zoom=True,
        frame_margins=None,
        alignment_offset=0,
    ):
        h, w = rgb.shape[:2]
        self.rgb = rgb
        same_size = self.size == (w, h)
        if not same_size or self.image_item is None:
            self.scene().clear()
            self.markers = []
            self.guide = None
            self.frame_guides = []
            self.image_item = self.scene().addPixmap(pixmap(rgb))
            self.size = (w, h)
            self.scene().setSceneRect(QRectF(-20, -20, w + 40, h + 40))
        else:
            self.image_item.setPixmap(pixmap(rgb))
        for marker in self.markers:
            self.scene().removeItem(marker)
        self.markers = []
        for name, point, color in markers or []:
            if point is None:
                continue
            pen = QPen(QColor(color), 2)
            pen.setCosmetic(True)
            circle = self.scene().addEllipse(-7, -7, 14, 14, pen)
            circle.setPos(*point)
            circle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label = self.scene().addSimpleText(name)
            label.setBrush(QColor(color))
            label.setPos(*point)
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            self.markers.extend([circle, label])
        if self.guide is not None:
            self.scene().removeItem(self.guide)
            self.guide = None
        for line in self.frame_guides:
            self.scene().removeItem(line)
        self.frame_guides = []
        if guide:
            pen = QPen(QColor("#24ac8a"), 1.2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self.guide = self.scene().addLine((w - 1) / 2, -0.5, (w - 1) / 2, h - 0.5, pen)
            if alignment_offset:
                offset_pen = QPen(QColor("#6cd6ff"), 1.2, Qt.PenStyle.DashLine)
                offset_pen.setCosmetic(True)
                axis = (w - 1) / 2 + alignment_offset
                self.frame_guides.append(
                    self.scene().addLine(axis, -0.5, axis, h - 0.5, offset_pen)
                )
            if frame_margins is not None:
                top, side, bottom = frame_margins
                boundary_pen = QPen(QColor("#d86b67"), 1.2, Qt.PenStyle.DashLine)
                boundary_pen.setCosmetic(True)
                for coordinates in [
                    (-0.5, top - 0.5, w - 0.5, top - 0.5),
                    (side - 0.5, -0.5, side - 0.5, h - 0.5),
                    (w - side - 0.5, -0.5, w - side - 0.5, h - 0.5),
                    (-0.5, h - bottom - 0.5, w - 0.5, h - bottom - 0.5),
                ]:
                    self.frame_guides.append(self.scene().addLine(*coordinates, boundary_pen))
        if not same_size or not preserve_zoom or self.auto_fit:
            self.fit()

    def fit(self):
        self.auto_fit = True
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor):
        current = self.transform().m11()
        if 0.015 <= current * factor <= 20:
            self.auto_fit = False
            self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit()

    def wheelEvent(self, event):
        self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space_tool = self.tool
            self.set_tool("pan")
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.space_tool is not None:
            tool, self.space_tool = self.space_tool, None
            self.set_tool(tool)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def valid_point(self, event):
        point = self.mapToScene(event.position().toPoint())
        if self.size is None or not (
            0 <= point.x() < self.size[0] and 0 <= point.y() < self.size[1]
        ):
            return None
        return point

    def mousePressEvent(self, event):
        point = self.valid_point(event)
        if event.button() == Qt.MouseButton.LeftButton and point is not None:
            if self.tool == "portrait":
                self.portrait_drag = (point, self.rgb)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.portrait_drag_started.emit()
                event.accept()
                return
            if self.tool in ("erase", "restore"):
                self.painting = True
                self.stroke_started.emit()
                self.brush_moved.emit(point.x(), point.y())
                event.accept()
                return
            if self.tool != "pan":
                self.point_selected.emit(point.x(), point.y())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.portrait_drag is not None:
            point = self.mapToScene(event.position().toPoint())
            start, _ = self.portrait_drag
            self.portrait_moved.emit(
                point.x() - start.x(),
                point.y() - start.y(),
                bool(event.modifiers() & Qt.KeyboardModifier.AltModifier),
            )
            event.accept()
            return
        if self.painting:
            point = self.valid_point(event)
            if point is not None:
                self.brush_moved.emit(point.x(), point.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.portrait_drag is not None:
            self.portrait_drag = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.portrait_drag_finished.emit()
            event.accept()
            return
        if self.painting:
            self.painting = False
            self.stroke_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def preview_translation(self, dx, dy):
        if self.portrait_drag is not None:
            self.image_item.setPixmap(pixmap(translate_layer(self.portrait_drag[1], dx, dy, 255)))
