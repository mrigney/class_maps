"""Image canvas: QGraphicsView with pan/zoom, click-to-label, and line drawing."""

import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPointF

from class_maps.gui.overlay_renderer import (
    numpy_to_qpixmap,
    render_superpixel_boundaries,
    render_class_overlay,
    render_density_overlay,
)


# Tool modes
MODE_LABEL = "label"
MODE_DRAW_LINE = "draw_line"


class ImageCanvas(QGraphicsView):
    """Interactive image display with pan, zoom, click-to-label, and line drawing.

    Signals
    -------
    pixel_clicked(int, int)
        Emitted when the user left-clicks on the image (row, col).
    pixel_right_clicked(int, int)
        Emitted when the user right-clicks on the image (row, col).
    cursor_moved(int, int)
        Emitted when the mouse moves over the image (row, col).
    polyline_finished(list)
        Emitted when the user finishes drawing a polyline.
        Payload is a list of (row, col) tuples.
    """

    pixel_clicked = pyqtSignal(int, int)
    pixel_right_clicked = pyqtSignal(int, int)
    cursor_moved = pyqtSignal(int, int)
    polyline_finished = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Rendering
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)

        # Image layers
        self._image_item = None
        self._boundary_item = None
        self._classification_item = None
        self._density_item = None
        self._label_feedback_item = None
        self._drawn_lines_item = None

        # State
        self._image_shape = None
        self._panning = False
        self._pan_start = QPointF()
        self._labels = None  # superpixel labels array

        # Tool mode
        self._mode = MODE_LABEL

        # Line drawing state
        self._drawing_points = []  # current polyline being drawn [(row, col), ...]
        self._drawing_items = []   # QGraphicsItems for current in-progress line
        self._drawn_polylines = [] # completed: [(points, width), ...] where points=[(row,col),...]
        self._line_width = 10      # current pixel width for new lines
        self._drawn_lines_overlay_items = []  # QGraphicsItems for completed lines

        # Overlay visibility
        self._show_boundaries = False
        self._show_classification = False
        self._show_density = False

    # --- Tool mode ---

    def set_mode(self, mode):
        """Set the interaction mode.

        Parameters
        ----------
        mode : str
            One of MODE_LABEL or MODE_DRAW_LINE.
        """
        if mode == self._mode:
            return
        # Cancel any in-progress drawing when switching away
        if self._mode == MODE_DRAW_LINE and self._drawing_points:
            self._cancel_drawing()
        self._mode = mode
        if mode == MODE_DRAW_LINE:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def get_mode(self):
        """Return the current interaction mode."""
        return self._mode

    def set_line_width(self, width):
        """Set the pixel width for new drawn linear features.

        Does not change the width of already-drawn lines.
        """
        self._line_width = max(1, width)

    def get_drawn_polylines(self):
        """Return all completed polylines as (points, width) tuples.

        Returns
        -------
        list of (list of (int, int), int)
            Each entry is (points, width) where points is [(row, col), ...].
        """
        return list(self._drawn_polylines)

    def set_drawn_polylines(self, polylines):
        """Set drawn polylines (e.g. when loading a profile).

        Parameters
        ----------
        polylines : list of (list of (int, int), int)
            Each entry is (points, width).
        """
        self._drawn_polylines = [(list(pts), w) for pts, w in polylines]
        self._redraw_all_lines()

    def clear_drawn_lines(self):
        """Remove all drawn polylines."""
        self._drawn_polylines = []
        self._cancel_drawing()
        self._redraw_all_lines()

    def undo_last_line(self):
        """Remove the most recently drawn polyline."""
        if self._drawn_polylines:
            self._drawn_polylines.pop()
            self._redraw_all_lines()

    def set_image(self, rgb_array):
        """Display an RGB image on the canvas.

        Parameters
        ----------
        rgb_array : np.ndarray
            (H, W, 3) uint8 RGB array.
        """
        self._scene.clear()
        self._image_shape = rgb_array.shape[:2]

        pixmap = numpy_to_qpixmap(rgb_array)
        self._image_item = self._scene.addPixmap(pixmap)
        self._image_item.setZValue(0)

        # Reset overlay items
        self._boundary_item = None
        self._classification_item = None
        self._density_item = None
        self._label_feedback_item = None
        self._drawn_lines_item = None
        self._drawing_items = []
        self._drawn_lines_overlay_items = []

        # Redraw any existing polylines
        self._redraw_all_lines()

        # Fit the image in view
        self.setSceneRect(self._scene.itemsBoundingRect())
        self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def set_superpixel_labels(self, labels):
        """Store the superpixel label array for click-to-label mapping.

        Parameters
        ----------
        labels : np.ndarray
            (H, W) int32 superpixel label array.
        """
        self._labels = labels

    def set_boundary_overlay(self, labels, visible=None):
        """Render and set the superpixel boundary overlay.

        Parameters
        ----------
        labels : np.ndarray
            (H, W) int32 superpixel label array.
        visible : bool, optional
            Whether to show the overlay. If None, uses current state.
        """
        if self._image_shape is None:
            return

        # Remove old boundary overlay
        if self._boundary_item is not None:
            self._scene.removeItem(self._boundary_item)
            self._boundary_item = None

        qimg = render_superpixel_boundaries(labels, self._image_shape)
        pixmap = QPixmap.fromImage(qimg)
        self._boundary_item = self._scene.addPixmap(pixmap)
        self._boundary_item.setZValue(10)

        if visible is not None:
            self._show_boundaries = visible
        self._boundary_item.setVisible(self._show_boundaries)

    def set_classification_overlay(self, class_raster, class_colors, alpha=128):
        """Render and set the classification overlay.

        Parameters
        ----------
        class_raster : np.ndarray
            (H, W) uint8 class ID array.
        class_colors : dict
            {class_id: (R, G, B)}.
        alpha : int
            Overlay alpha (0-255).
        """
        if self._classification_item is not None:
            self._scene.removeItem(self._classification_item)
            self._classification_item = None

        qimg = render_class_overlay(class_raster, class_colors, alpha)
        pixmap = QPixmap.fromImage(qimg)
        self._classification_item = self._scene.addPixmap(pixmap)
        self._classification_item.setZValue(20)
        self._classification_item.setVisible(self._show_classification)

    def set_density_overlay(self, density_raster, alpha=128):
        """Render and set the density overlay.

        Parameters
        ----------
        density_raster : np.ndarray
            (H, W) float32 density in [0, 1].
        alpha : int
            Maximum overlay alpha.
        """
        if self._density_item is not None:
            self._scene.removeItem(self._density_item)
            self._density_item = None

        qimg = render_density_overlay(density_raster, alpha)
        pixmap = QPixmap.fromImage(qimg)
        self._density_item = self._scene.addPixmap(pixmap)
        self._density_item.setZValue(25)
        self._density_item.setVisible(self._show_density)

    def set_label_feedback(self, labels, labeled_segments, class_colors, alpha=100):
        """Render labeled superpixel feedback overlay.

        Shows which superpixels the user has labeled, colored by class.

        Parameters
        ----------
        labels : np.ndarray
            (H, W) int32 superpixel label array.
        labeled_segments : dict
            {segment_id: class_id}.
        class_colors : dict
            {class_id: (R, G, B)}.
        alpha : int
            Overlay alpha.
        """
        if self._label_feedback_item is not None:
            self._scene.removeItem(self._label_feedback_item)
            self._label_feedback_item = None

        if not labeled_segments:
            return

        h, w = labels.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        for seg_id, class_id in labeled_segments.items():
            if class_id not in class_colors:
                continue
            mask = labels == seg_id
            color = class_colors[class_id]
            overlay[mask, 0] = color[0]
            overlay[mask, 1] = color[1]
            overlay[mask, 2] = color[2]
            overlay[mask, 3] = alpha

        from class_maps.gui.overlay_renderer import numpy_to_qimage
        pixmap = QPixmap.fromImage(numpy_to_qimage(overlay))
        self._label_feedback_item = self._scene.addPixmap(pixmap)
        self._label_feedback_item.setZValue(15)

    def toggle_boundaries(self, visible):
        """Show or hide the superpixel boundary overlay."""
        self._show_boundaries = visible
        if self._boundary_item is not None:
            self._boundary_item.setVisible(visible)

    def toggle_classification(self, visible):
        """Show or hide the classification overlay."""
        self._show_classification = visible
        if self._classification_item is not None:
            self._classification_item.setVisible(visible)

    def toggle_density(self, visible):
        """Show or hide the density overlay."""
        self._show_density = visible
        if self._density_item is not None:
            self._density_item.setVisible(visible)

    # --- Mouse events ---

    def wheelEvent(self, event):
        """Zoom in/out with scroll wheel."""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1.0 / factor, 1.0 / factor)

    def mousePressEvent(self, event):
        """Handle pan, label clicks, and line drawing depending on mode."""
        # Pan: middle button or Ctrl+left always works regardless of mode
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if self._mode == MODE_DRAW_LINE:
            self._handle_draw_press(event)
        else:
            self._handle_label_press(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to finish a polyline in draw mode."""
        if self._mode == MODE_DRAW_LINE and event.button() == Qt.LeftButton:
            self._finish_drawing()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """Handle panning, cursor tracking, and live line preview."""
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
        else:
            pos = self._map_to_image(event.pos())
            if pos is not None:
                self.cursor_moved.emit(pos[0], pos[1])
                # Live preview of line segment being drawn
                if (self._mode == MODE_DRAW_LINE and self._drawing_points):
                    self._update_drawing_preview(pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End panning."""
        if event.button() in (Qt.MiddleButton, Qt.LeftButton) and self._panning:
            self._panning = False
            if self._mode == MODE_DRAW_LINE:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Handle Escape to cancel drawing, Ctrl+Z to undo last line."""
        if self._mode == MODE_DRAW_LINE:
            if event.key() == Qt.Key_Escape:
                self._cancel_drawing()
                event.accept()
                return
            if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
                if self._drawing_points:
                    # Undo last point in current drawing
                    self._drawing_points.pop()
                    self._redraw_in_progress_line()
                else:
                    self.undo_last_line()
                event.accept()
                return
        super().keyPressEvent(event)

    # --- Label mode handlers ---

    def _handle_label_press(self, event):
        """Handle clicks in label mode."""
        if event.button() == Qt.LeftButton:
            pos = self._map_to_image(event.pos())
            if pos is not None:
                self.pixel_clicked.emit(pos[0], pos[1])
            event.accept()
        elif event.button() == Qt.RightButton:
            pos = self._map_to_image(event.pos())
            if pos is not None:
                self.pixel_right_clicked.emit(pos[0], pos[1])
            event.accept()
        else:
            super().mousePressEvent(event)

    # --- Draw mode handlers ---

    def _handle_draw_press(self, event):
        """Handle clicks in line drawing mode."""
        if event.button() == Qt.LeftButton:
            pos = self._map_to_image(event.pos())
            if pos is not None:
                self._drawing_points.append(pos)
                self._redraw_in_progress_line()
            event.accept()
        elif event.button() == Qt.RightButton:
            # Right-click finishes the polyline (alternative to double-click)
            if self._drawing_points:
                self._finish_drawing()
            event.accept()

    def _finish_drawing(self):
        """Finish the current polyline and emit signal."""
        if len(self._drawing_points) < 2:
            self._cancel_drawing()
            return
        polyline = list(self._drawing_points)
        self._drawn_polylines.append((polyline, self._line_width))
        self._drawing_points = []
        self._clear_in_progress_items()
        self._redraw_all_lines()
        self.polyline_finished.emit(polyline)

    def _cancel_drawing(self):
        """Cancel the current in-progress polyline."""
        self._drawing_points = []
        self._clear_in_progress_items()

    def _clear_in_progress_items(self):
        """Remove in-progress drawing items from the scene."""
        for item in self._drawing_items:
            self._scene.removeItem(item)
        self._drawing_items = []

    def _redraw_in_progress_line(self):
        """Redraw the in-progress polyline on the scene."""
        self._clear_in_progress_items()
        if len(self._drawing_points) < 1:
            return

        pen = QPen(QColor(255, 100, 0, 200), max(1, self._line_width), Qt.SolidLine,
                   Qt.RoundCap, Qt.RoundJoin)

        # Draw line segments between points
        for i in range(len(self._drawing_points) - 1):
            r1, c1 = self._drawing_points[i]
            r2, c2 = self._drawing_points[i + 1]
            item = self._scene.addLine(c1, r1, c2, r2, pen)
            item.setZValue(30)
            self._drawing_items.append(item)

        # Draw dots at vertices
        dot_pen = QPen(QColor(255, 255, 0, 220))
        dot_pen.setWidth(0)
        for r, c in self._drawing_points:
            dot_r = max(3, self._line_width / 3)
            item = self._scene.addEllipse(
                c - dot_r, r - dot_r, dot_r * 2, dot_r * 2,
                dot_pen, QColor(255, 255, 0, 180)
            )
            item.setZValue(31)
            self._drawing_items.append(item)

    def _update_drawing_preview(self, cursor_pos):
        """Update the live preview segment from last point to cursor."""
        # Remove previous preview line (last item if it's a preview)
        if (hasattr(self, "_preview_item") and self._preview_item is not None):
            self._scene.removeItem(self._preview_item)
            self._preview_item = None

        if not self._drawing_points:
            return

        last_r, last_c = self._drawing_points[-1]
        cur_r, cur_c = cursor_pos

        pen = QPen(QColor(255, 100, 0, 100), max(1, self._line_width),
                   Qt.DashLine, Qt.RoundCap, Qt.RoundJoin)
        self._preview_item = self._scene.addLine(last_c, last_r, cur_c, cur_r, pen)
        self._preview_item.setZValue(30)

    def _redraw_all_lines(self):
        """Redraw all completed polylines on the scene, each at its own width."""
        # Remove old completed line items
        for item in self._drawn_lines_overlay_items:
            self._scene.removeItem(item)
        self._drawn_lines_overlay_items = []

        if not self._drawn_polylines:
            return

        for polyline, width in self._drawn_polylines:
            pen = QPen(QColor(255, 50, 0, 180), max(1, width),
                       Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            for i in range(len(polyline) - 1):
                r1, c1 = polyline[i]
                r2, c2 = polyline[i + 1]
                item = self._scene.addLine(c1, r1, c2, r2, pen)
                item.setZValue(28)
                self._drawn_lines_overlay_items.append(item)

    # --- Coordinate mapping ---

    def _map_to_image(self, view_pos):
        """Map a view coordinate to image pixel (row, col).

        Returns None if outside the image bounds.
        """
        scene_pos = self.mapToScene(view_pos)
        col = int(scene_pos.x())
        row = int(scene_pos.y())

        if self._image_shape is None:
            return None
        h, w = self._image_shape
        if 0 <= row < h and 0 <= col < w:
            return (row, col)
        return None
