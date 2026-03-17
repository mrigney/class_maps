"""Overlay rendering: classification, density, and superpixel boundary overlays."""

import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def numpy_to_qimage(array):
    """Convert a numpy array to QImage.

    Parameters
    ----------
    array : np.ndarray
        (H, W, 4) uint8 RGBA array, or (H, W, 3) uint8 RGB array.

    Returns
    -------
    QImage
    """
    array = np.ascontiguousarray(array)
    h, w = array.shape[:2]

    if array.ndim == 3 and array.shape[2] == 4:
        fmt = QImage.Format_RGBA8888
        bytes_per_line = 4 * w
    elif array.ndim == 3 and array.shape[2] == 3:
        fmt = QImage.Format_RGB888
        bytes_per_line = 3 * w
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

    qimg = QImage(array.data, w, h, bytes_per_line, fmt)
    # QImage doesn't own the data, so we need to copy
    return qimg.copy()


def numpy_to_qpixmap(array):
    """Convert a numpy array to QPixmap."""
    return QPixmap.fromImage(numpy_to_qimage(array))


def render_superpixel_boundaries(labels, shape, color=(255, 255, 0), alpha=180):
    """Render superpixel boundaries as a transparent RGBA overlay.

    Parameters
    ----------
    labels : np.ndarray
        (H, W) int32 superpixel label array.
    shape : tuple
        (H, W) of the output image.
    color : tuple
        (R, G, B) boundary color.
    alpha : int
        Alpha value for boundary pixels (0-255).

    Returns
    -------
    QImage
        RGBA QImage with boundary lines.
    """
    h, w = shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    # Find boundary pixels: pixels where any neighbor has a different label
    boundary = np.zeros((h, w), dtype=bool)

    # Check horizontal neighbors
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]

    # Check vertical neighbors
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]

    overlay[boundary, 0] = color[0]
    overlay[boundary, 1] = color[1]
    overlay[boundary, 2] = color[2]
    overlay[boundary, 3] = alpha

    return numpy_to_qimage(overlay)


def render_class_overlay(class_raster, class_colors, alpha=128):
    """Render a classification overlay as a transparent RGBA image.

    Parameters
    ----------
    class_raster : np.ndarray
        (H, W) uint8 array of class IDs.
    class_colors : dict
        {class_id: (R, G, B)} color mapping.
    alpha : int
        Alpha value for classified pixels (0-255).

    Returns
    -------
    QImage
        RGBA QImage with classification colors.
    """
    h, w = class_raster.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    for class_id, color in class_colors.items():
        mask = class_raster == class_id
        if not np.any(mask):
            continue
        overlay[mask, 0] = color[0]
        overlay[mask, 1] = color[1]
        overlay[mask, 2] = color[2]
        overlay[mask, 3] = alpha

    return numpy_to_qimage(overlay)


def render_density_overlay(density_raster, alpha=128):
    """Render a density overlay as a heatmap (transparent RGBA).

    Uses a green-to-red colormap: 0.0 = transparent, 1.0 = bright green.

    Parameters
    ----------
    density_raster : np.ndarray
        (H, W) float32 array in [0.0, 1.0].
    alpha : int
        Maximum alpha value (0-255).

    Returns
    -------
    QImage
        RGBA QImage heatmap overlay.
    """
    h, w = density_raster.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    # Only show density where > 0
    mask = density_raster > 0.01
    if not np.any(mask):
        return numpy_to_qimage(overlay)

    density_clipped = np.clip(density_raster, 0.0, 1.0)

    # Green channel scales with density
    overlay[mask, 0] = 0
    overlay[mask, 1] = (density_clipped[mask] * 255).astype(np.uint8)
    overlay[mask, 2] = 0
    overlay[mask, 3] = (density_clipped[mask] * alpha).astype(np.uint8)

    return numpy_to_qimage(overlay)
