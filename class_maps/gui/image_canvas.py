"""Image canvas: QGraphicsView with pan/zoom and click-to-label support."""

import numpy as np
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt, pyqtSignal, QPointF

from class_maps.gui.overlay_renderer import (
    numpy_to_qpixmap,
    render_superpixel_boundaries,
    render_class_overlay,
    render_density_overlay,
)


class ImageCanvas(QGraphicsView):
    """Interactive image display with pan, zoom, and click-to-label.

    Signals
    -------
    pixel_clicked(int, int)
        Emitted when the user left-clicks on the image (row, col).
    pixel_right_clicked(int, int)
        Emitted when the user right-clicks on the image (row, col).
    cursor_moved(int, int)
        Emitted when the mouse moves over the image (row, col).
    """

    pixel_clicked = pyqtSignal(int, int)
    pixel_right_clicked = pyqtSignal(int, int)
    cursor_moved = pyqtSignal(int, int)

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

        # State
        self._image_shape = None
        self._panning = False
        self._pan_start = QPointF()
        self._labels = None  # superpixel labels array

        # Overlay visibility
        self._show_boundaries = False
        self._show_classification = False
        self._show_density = False

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
        """Handle pan (middle button) and label clicks (left/right)."""
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            # Ctrl+left click = pan
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.LeftButton:
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

    def mouseMoveEvent(self, event):
        """Handle panning and cursor position tracking."""
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
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End panning."""
        if event.button() in (Qt.MiddleButton, Qt.LeftButton) and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

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
