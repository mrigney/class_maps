"""Linear feature pre-detection for roads, rivers, and other narrow elongated structures.

Roads and rivers are poorly served by SLIC superpixels because SLIC optimizes for
compact, roughly equal-sized regions. A 10-pixel-wide road gets absorbed into a
surrounding 50x50 superpixel, mixing road pixels with adjacent terrain.

This module detects linear features BEFORE SLIC runs, carves them out as their own
segments, and lets SLIC operate only on the remaining pixels.

Detection strategy:
- PRIMARY (U-Net): When trained model weights are available, uses a U-Net
  (ResNet34 encoder) for semantic road segmentation. Handles all road types,
  orientations, and widths with high accuracy.
- FALLBACK (heuristic): When no model is available, uses local uniformity +
  edge detection. Less accurate but requires no training.
"""

import numpy as np
import cv2
from scipy import ndimage


# Default parameters (used by heuristic fallback)
DEFAULT_ROAD_WIDTHS = (5, 9, 15, 25)  # pixels — expected road widths
DEFAULT_UNIFORMITY_THRESHOLD = 15.0  # max local std to be "uniform"
DEFAULT_EDGE_LOW = 20     # Canny low threshold
DEFAULT_EDGE_HIGH = 80    # Canny high threshold
DEFAULT_MIN_LENGTH = 40  # minimum linear segment length in pixels
DEFAULT_MAX_WIDTH = 40   # maximum width to be considered linear
DEFAULT_MIN_ASPECT = 3.5  # minimum length/width ratio

# U-Net default confidence threshold
DEFAULT_UNET_CONFIDENCE = 0.5

# Cached model reference (loaded once, reused)
_cached_model = None
_model_checked = False


def detect_linear_features(gray, rgb, road_widths=None,
                           uniformity_threshold=None,
                           min_length=None, max_width=None,
                           ridge_threshold=None,  # kept for API compat
                           color_var_threshold=None,  # kept for API compat
                           confidence=None,
                           use_unet=True,
                           progress_callback=None):
    """Detect linear features (roads, rivers, paths) in a satellite image.

    Uses a trained U-Net model when available, otherwise falls back to
    heuristic detection.

    Parameters
    ----------
    gray : np.ndarray
        (H, W) uint8 grayscale image.
    rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    road_widths : tuple of int, optional
        Expected road widths in pixels (heuristic only).
    uniformity_threshold : float, optional
        Maximum local std dev for "uniform" pixels (heuristic only).
    min_length : int, optional
        Minimum length of a linear segment to keep.
    max_width : int, optional
        Maximum average width for a feature to be "linear."
    confidence : float, optional
        U-Net probability threshold (0.0-1.0). Default 0.5.
    use_unet : bool
        Whether to attempt U-Net detection. Set False to force heuristic.
    progress_callback : callable, optional
        Called with (current, total) for progress reporting (U-Net only).

    Returns
    -------
    np.ndarray
        (H, W) bool mask of detected linear feature pixels.
    """
    if min_length is None:
        min_length = DEFAULT_MIN_LENGTH
    if max_width is None:
        max_width = DEFAULT_MAX_WIDTH
    if confidence is None:
        confidence = DEFAULT_UNET_CONFIDENCE

    # Try U-Net detection first
    if use_unet:
        unet_mask = _detect_with_unet(rgb, confidence, progress_callback)
        if unet_mask is not None:
            # Apply geometry filter to U-Net output to remove noise
            unet_mask = _filter_by_geometry(unet_mask, min_length, max_width)
            return unet_mask

    # Fall back to heuristic detection
    return _detect_heuristic(gray, rgb, road_widths, uniformity_threshold,
                             min_length, max_width)


def has_unet_model():
    """Check whether trained U-Net weights are available.

    Returns
    -------
    bool
        True if a trained model can be loaded.
    """
    try:
        from class_maps.core.road_model import get_weight_path
        return get_weight_path() is not None
    except ImportError:
        return False


def _detect_with_unet(rgb, confidence, progress_callback):
    """Attempt road detection using the U-Net model.

    Returns
    -------
    np.ndarray or None
        (H, W) bool mask, or None if model not available.
    """
    global _cached_model, _model_checked

    try:
        from class_maps.core.road_model import load_model, predict_mask
    except ImportError:
        _model_checked = True
        return None

    # Load model (cached after first load)
    if _cached_model is None:
        if _model_checked:
            return None
        _model_checked = True
        _cached_model = load_model()
        if _cached_model is None:
            return None

    return predict_mask(_cached_model, rgb, confidence=confidence,
                        progress_callback=progress_callback)


def _detect_heuristic(gray, rgb, road_widths, uniformity_threshold,
                      min_length, max_width):
    """Heuristic linear feature detection using uniformity + edge pairs.

    Used as fallback when no U-Net model is available.
    """
    if road_widths is None:
        road_widths = DEFAULT_ROAD_WIDTHS
    if uniformity_threshold is None:
        uniformity_threshold = DEFAULT_UNIFORMITY_THRESHOLD

    # Step 1: Find locally uniform regions
    uniform_mask = _find_uniform_regions(gray, road_widths, uniformity_threshold)

    # Step 2: Find regions near parallel edges
    edge_pair_mask = _detect_edge_pairs(gray, road_widths)

    # Step 3: Intersection
    candidates = uniform_mask & edge_pair_mask

    # Step 4: Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidates = cv2.morphologyEx(candidates.astype(np.uint8),
                                  cv2.MORPH_CLOSE, kernel_close).astype(bool)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidates = cv2.morphologyEx(candidates.astype(np.uint8),
                                  cv2.MORPH_OPEN, kernel_open).astype(bool)

    # Step 5: Exclude shadows
    candidates &= gray.astype(np.float32) > 50

    # Step 6: Filter by geometry
    candidates = _filter_by_geometry(candidates, min_length, max_width)

    # Step 7: Color uniformity check
    candidates = _filter_by_color_uniformity(candidates, rgb)

    return candidates


def _find_uniform_regions(gray, road_widths, threshold):
    """Find pixels where the local grayscale standard deviation is low."""
    gray_f = gray.astype(np.float32)
    uniform = np.zeros(gray.shape, dtype=bool)

    for width in road_widths:
        win = max(3, width)
        if win % 2 == 0:
            win += 1

        local_mean = cv2.blur(gray_f, (win, win))
        local_sq_mean = cv2.blur(gray_f ** 2, (win, win))
        local_var = np.clip(local_sq_mean - local_mean ** 2, 0, None)
        local_std = np.sqrt(local_var)

        uniform |= local_std < threshold

    return uniform


def _detect_edge_pairs(gray, road_widths):
    """Detect pixels between parallel edges at road-width spacing."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, DEFAULT_EDGE_LOW, DEFAULT_EDGE_HIGH)
    edge_bool = edges > 0

    h, w = gray.shape
    result = np.zeros((h, w), dtype=bool)

    for width in road_widths:
        half_w = max(2, width // 2)

        # Vertical direction (horizontal roads)
        edge_above = np.zeros_like(edge_bool)
        edge_below = np.zeros_like(edge_bool)
        for d in range(max(1, half_w - 2), min(half_w + 3, h)):
            edge_above[d:, :] |= edge_bool[:-d, :]
            edge_below[:-d, :] |= edge_bool[d:, :]
        between_h = edge_above & edge_below

        # Horizontal direction (vertical roads)
        edge_left = np.zeros_like(edge_bool)
        edge_right = np.zeros_like(edge_bool)
        for d in range(max(1, half_w - 2), min(half_w + 3, w)):
            edge_left[:, d:] |= edge_bool[:, :-d]
            edge_right[:, :-d] |= edge_bool[:, d:]
        between_v = edge_left & edge_right

        result |= between_h | between_v

    return result


def _find_uniform_regions(gray, road_widths, threshold):
    """Find pixels where the local grayscale standard deviation is low.

    Roads, paths, and water surfaces have low local texture variance compared
    to forest canopy, shrubs, or mixed terrain.

    Parameters
    ----------
    gray : np.ndarray
        (H, W) uint8 grayscale image.
    road_widths : tuple of int
        Window sizes for local std computation.
    threshold : float
        Maximum local std to be considered "uniform."

    Returns
    -------
    np.ndarray
        (H, W) bool mask of uniform pixels.
    """
    gray_f = gray.astype(np.float32)
    uniform = np.zeros(gray.shape, dtype=bool)

    for width in road_widths:
        # Window size should be slightly larger than road width to capture
        # the road surface without too much border contamination
        win = max(3, width)
        if win % 2 == 0:
            win += 1

        # Local mean and variance
        local_mean = cv2.blur(gray_f, (win, win))
        local_sq_mean = cv2.blur(gray_f ** 2, (win, win))
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.clip(local_var, 0, None)
        local_std = np.sqrt(local_var)

        # Pixels with low local std at this scale are uniform
        uniform |= local_std < threshold

    return uniform


def _filter_by_geometry(mask, min_length, max_width):
    """Remove detected regions that are too short, too wide, or not elongated.

    Parameters
    ----------
    mask : np.ndarray
        (H, W) bool candidate mask.
    min_length : int
        Minimum length of a connected component.
    max_width : int
        Maximum average width.

    Returns
    -------
    np.ndarray
        (H, W) bool filtered mask.
    """
    labeled, n_components = ndimage.label(mask)
    result = np.zeros_like(mask)

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        area = comp_mask.sum()

        if area < min_length:
            continue

        # Bounding box dimensions
        rows, cols = np.where(comp_mask)
        height = rows.max() - rows.min() + 1
        width_bb = cols.max() - cols.min() + 1

        # Length = longer dimension, width = area / length
        length = max(height, width_bb)
        avg_width = area / max(length, 1)

        if length < min_length:
            continue

        if avg_width > max_width:
            continue

        # Aspect ratio: linear features should be elongated
        aspect = length / max(avg_width, 1)
        if aspect < DEFAULT_MIN_ASPECT:
            continue

        result[comp_mask] = True

    return result


def _filter_by_color_uniformity(mask, rgb, max_channel_std=25.0):
    """Remove detected regions with high per-channel color variation.

    Roads tend to be a single material with low color variance.

    Parameters
    ----------
    mask : np.ndarray
        (H, W) bool candidate mask.
    rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    max_channel_std : float
        Maximum per-channel std dev allowed.

    Returns
    -------
    np.ndarray
        (H, W) bool filtered mask.
    """
    labeled, n_components = ndimage.label(mask)
    result = np.zeros_like(mask)

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        pixels = rgb[comp_mask].astype(np.float64)

        # Check per-channel std
        channel_stds = pixels.std(axis=0)
        if channel_stds.max() <= max_channel_std:
            result[comp_mask] = True

    return result


def linear_mask_to_segments(mask):
    """Convert a linear feature mask to labeled segments.

    Each connected component in the mask becomes its own segment.

    Parameters
    ----------
    mask : np.ndarray
        (H, W) bool linear feature mask.

    Returns
    -------
    np.ndarray
        (H, W) int32 label array. 0 = not a linear feature.
    int
        Number of linear segments found.
    """
    labeled, n_components = ndimage.label(mask)
    return labeled.astype(np.int32), n_components


def merge_linear_and_slic(linear_labels, n_linear, slic_labels):
    """Merge linear feature segments with SLIC superpixel segments.

    Linear segments take priority — where a linear segment exists, it overrides
    the SLIC label. SLIC segment IDs are offset to avoid collisions.

    Parameters
    ----------
    linear_labels : np.ndarray
        (H, W) int32 linear feature labels (0 = no linear feature).
    n_linear : int
        Number of linear segments.
    slic_labels : np.ndarray
        (H, W) int32 SLIC superpixel labels.

    Returns
    -------
    np.ndarray
        (H, W) int32 merged label array.
    set of int
        Set of segment IDs that are linear features.
    """
    slic_offset = n_linear + 1
    merged = slic_labels.copy() + slic_offset

    linear_mask = linear_labels > 0
    merged[linear_mask] = linear_labels[linear_mask]

    linear_segment_ids = set(range(1, n_linear + 1))

    return merged.astype(np.int32), linear_segment_ids
