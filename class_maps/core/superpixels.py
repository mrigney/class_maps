"""SLIC superpixel segmentation and segment property computation.

Supports optional linear feature pre-detection: roads, rivers, and paths are
detected first and carved out as their own segments before SLIC runs on the
remaining pixels. This prevents narrow linear features from being absorbed
into surrounding terrain superpixels.
"""

import numpy as np
from skimage.segmentation import slic, mark_boundaries

from class_maps.config import SLIC_N_SEGMENTS, SLIC_COMPACTNESS, SLIC_SIGMA


def compute_slic(rgb_array, n_segments=None, compactness=None, sigma=None):
    """Compute SLIC superpixel segmentation.

    Parameters
    ----------
    rgb_array : np.ndarray
        (H, W, 3) uint8 RGB image.
    n_segments : int, optional
        Approximate number of superpixels. Default from config.
    compactness : float, optional
        Balance between color and spatial proximity. Default from config.
    sigma : float, optional
        Gaussian smoothing sigma. Default from config.

    Returns
    -------
    np.ndarray
        (H, W) int32 label array where each pixel has its segment ID.
    """
    if n_segments is None:
        n_segments = SLIC_N_SEGMENTS
    if compactness is None:
        compactness = SLIC_COMPACTNESS
    if sigma is None:
        sigma = SLIC_SIGMA

    labels = slic(
        rgb_array,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
    )
    return labels.astype(np.int32)


def compute_slic_with_linear(rgb_array, gray, n_segments=None, compactness=None,
                              sigma=None, detect_linear=True, linear_params=None):
    """Compute SLIC superpixels with optional linear feature pre-detection.

    When detect_linear=True, roads/rivers/paths are detected first and carved
    out as dedicated segments. SLIC then runs only on the remaining pixels.
    This produces much better segmentation of narrow linear features.

    Parameters
    ----------
    rgb_array : np.ndarray
        (H, W, 3) uint8 RGB image.
    gray : np.ndarray
        (H, W) uint8 grayscale image.
    n_segments : int, optional
        Approximate number of SLIC superpixels.
    compactness : float, optional
        SLIC compactness parameter.
    sigma : float, optional
        SLIC smoothing sigma.
    detect_linear : bool
        Whether to run linear feature pre-detection.
    linear_params : dict, optional
        Parameters for linear feature detection. Keys:
        - road_widths: tuple of int
        - ridge_threshold: float
        - min_length: int
        - max_width: int
        - color_var_threshold: float

    Returns
    -------
    np.ndarray
        (H, W) int32 merged label array.
    set of int
        Set of segment IDs that are linear features (empty if detect_linear=False).
    """
    if not detect_linear:
        labels = compute_slic(rgb_array, n_segments, compactness, sigma)
        return labels, set()

    from class_maps.core.linear_features import (
        detect_linear_features, linear_mask_to_segments, merge_linear_and_slic,
    )

    # Step 1: Detect linear features
    params = linear_params or {}
    linear_mask = detect_linear_features(gray, rgb_array, **params)

    # Step 2: Convert to labeled segments
    linear_labels, n_linear = linear_mask_to_segments(linear_mask)

    if n_linear == 0:
        # No linear features detected, fall back to standard SLIC
        labels = compute_slic(rgb_array, n_segments, compactness, sigma)
        return labels, set()

    # Step 3: Run SLIC on non-linear pixels
    # Create a modified image where linear pixels are replaced with neighbor colors
    # so SLIC doesn't try to incorporate them
    slic_image = rgb_array.copy()

    # Replace linear pixels with local median of non-linear neighbors
    # This prevents SLIC from being attracted to road colors
    linear_bool = linear_mask
    if linear_bool.any():
        for ch in range(3):
            channel = slic_image[:, :, ch].astype(np.float32)
            # Dilate the non-linear region inward to fill linear pixels
            from scipy.ndimage import median_filter
            filled = median_filter(channel, size=7)
            channel[linear_bool] = filled[linear_bool]
            slic_image[:, :, ch] = channel.astype(np.uint8)

    slic_labels = compute_slic(slic_image, n_segments, compactness, sigma)

    # Step 4: Merge linear and SLIC segments
    merged, linear_ids = merge_linear_and_slic(linear_labels, n_linear, slic_labels)

    return merged, linear_ids


def get_segment_mask(labels, segment_id):
    """Get a boolean mask for a specific segment."""
    return labels == segment_id


def get_segment_ids(labels):
    """Get sorted list of unique segment IDs."""
    return np.unique(labels)


def get_boundary_image(rgb_array, labels, color=(255, 255, 0)):
    """Generate an image with superpixel boundaries drawn."""
    color_normalized = tuple(c / 255.0 for c in color)
    bounded = mark_boundaries(
        rgb_array.astype(np.float64) / 255.0,
        labels,
        color=color_normalized,
        mode="outer",
    )
    return (bounded * 255).clip(0, 255).astype(np.uint8)


def get_segment_properties(labels):
    """Compute basic properties for each segment.

    Returns
    -------
    dict
        {segment_id: {"area": int, "centroid": (row, col), "bbox": (r0, c0, r1, c1)}}
    """
    segment_ids = get_segment_ids(labels)
    properties = {}

    for sid in segment_ids:
        mask = labels == sid
        rows, cols = np.where(mask)

        area = len(rows)
        centroid = (float(rows.mean()), float(cols.mean()))
        bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

        properties[int(sid)] = {
            "area": area,
            "centroid": centroid,
            "bbox": bbox,
        }

    return properties
