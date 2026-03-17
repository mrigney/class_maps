"""Per-superpixel feature extraction: color, texture, spectral, edge, shape."""

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from class_maps.config import GLCM_DISTANCES, GLCM_ANGLES
from class_maps.utils.geometry_utils import (
    compute_perimeter, compute_compactness, compute_eccentricity,
)


# Feature names for debugging and importance analysis
FEATURE_NAMES = [
    # Color RGB (6)
    "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
    # Color HSV (6)
    "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std",
    # Color Lab (6)
    "l_mean", "l_std", "a_mean", "a_std", "b_lab_mean", "b_lab_std",
    # NDVI (2) - zeros if no NIR
    "ndvi_mean", "ndvi_std",
    # GLCM texture (6)
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation", "glcm_entropy",
    # Edge density (1)
    "edge_density",
    # Shape (3)
    "area_normalized", "compactness", "eccentricity",
]

N_FEATURES = len(FEATURE_NAMES)  # 30


def _channel_stats(pixels):
    """Compute mean and std for a 1D array of pixel values."""
    if len(pixels) == 0:
        return 0.0, 0.0
    return float(np.mean(pixels)), float(np.std(pixels))


def _compute_glcm_features(gray_patch, mask_patch):
    """Compute GLCM texture features for a masked gray patch.

    Parameters
    ----------
    gray_patch : np.ndarray
        (h, w) uint8 grayscale patch (bounding box of segment).
    mask_patch : np.ndarray
        (h, w) boolean mask within the bounding box.

    Returns
    -------
    list of float
        [contrast, dissimilarity, homogeneity, energy, correlation, entropy]
    """
    # Apply mask: set non-segment pixels to 0
    masked = gray_patch.copy()
    masked[~mask_patch] = 0

    # Quantize to 64 levels to keep GLCM tractable
    quantized = (masked / 4).astype(np.uint8)  # 256/4 = 64 levels

    if quantized.size < 4 or mask_patch.sum() < 4:
        return [0.0] * 6

    try:
        glcm = graycomatrix(
            quantized,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=64,
            symmetric=True,
            normed=True,
        )

        contrast = graycoprops(glcm, "contrast").mean()
        dissimilarity = graycoprops(glcm, "dissimilarity").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        energy = graycoprops(glcm, "energy").mean()
        correlation = graycoprops(glcm, "correlation").mean()

        # Entropy: -sum(p * log(p))
        # Compute from the GLCM directly
        glcm_norm = glcm.mean(axis=(2, 3))  # average over distances and angles
        glcm_norm = glcm_norm / (glcm_norm.sum() + 1e-10)
        entropy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))

        return [
            float(contrast), float(dissimilarity), float(homogeneity),
            float(energy), float(correlation), float(entropy),
        ]
    except Exception:
        return [0.0] * 6


def extract_segment_features(
    segment_id, labels, rgb, hsv, lab, gray, edges_canny, ndvi=None,
    total_pixels=None,
):
    """Extract the full feature vector for a single superpixel segment.

    Parameters
    ----------
    segment_id : int
        The segment ID.
    labels : np.ndarray
        (H, W) int32 superpixel label array.
    rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    hsv : np.ndarray
        (H, W, 3) uint8 HSV image.
    lab : np.ndarray
        (H, W, 3) uint8 Lab image.
    gray : np.ndarray
        (H, W) uint8 grayscale image.
    edges_canny : np.ndarray
        (H, W) uint8 binary edge map.
    ndvi : np.ndarray or None
        (H, W) float32 NDVI, or None if no NIR.
    total_pixels : int or None
        Total image pixels (for area normalization). Computed if None.

    Returns
    -------
    np.ndarray
        1D feature vector of length N_FEATURES.
    """
    mask = labels == segment_id
    rows, cols = np.where(mask)

    if len(rows) == 0:
        return np.zeros(N_FEATURES, dtype=np.float64)

    if total_pixels is None:
        total_pixels = labels.size

    features = []

    # --- Color features (18 dims) ---
    # RGB
    for ch in range(3):
        m, s = _channel_stats(rgb[mask, ch])
        features.extend([m, s])
    # HSV
    for ch in range(3):
        m, s = _channel_stats(hsv[mask, ch])
        features.extend([m, s])
    # Lab
    for ch in range(3):
        m, s = _channel_stats(lab[mask, ch])
        features.extend([m, s])

    # --- NDVI features (2 dims) ---
    if ndvi is not None:
        ndvi_m, ndvi_s = _channel_stats(ndvi[mask])
        features.extend([ndvi_m, ndvi_s])
    else:
        features.extend([0.0, 0.0])

    # --- GLCM texture features (6 dims) ---
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()) + 1, int(cols.max()) + 1
    gray_patch = gray[r0:r1, c0:c1]
    mask_patch = mask[r0:r1, c0:c1]
    glcm_feats = _compute_glcm_features(gray_patch, mask_patch)
    features.extend(glcm_feats)

    # --- Edge density (1 dim) ---
    edge_pixels = edges_canny[mask]
    edge_density = float(np.sum(edge_pixels > 0)) / max(len(edge_pixels), 1)
    features.append(edge_density)

    # --- Shape features (3 dims) ---
    area = len(rows)
    area_normalized = area / total_pixels
    features.append(area_normalized)

    perimeter = compute_perimeter(mask)
    compact = compute_compactness(area, perimeter)
    features.append(compact)

    eccent = compute_eccentricity(mask)
    features.append(eccent)

    return np.array(features, dtype=np.float64)


def extract_all_features(labels, rgb, hsv, lab, gray, edges_canny, ndvi=None,
                         progress_callback=None):
    """Extract features for all superpixel segments.

    Parameters
    ----------
    labels : np.ndarray
        (H, W) int32 superpixel label array.
    rgb, hsv, lab, gray, edges_canny :
        Preprocessed image arrays.
    ndvi : np.ndarray or None
        NDVI array or None.
    progress_callback : callable, optional
        Called with (current, total) for progress reporting.

    Returns
    -------
    np.ndarray
        (n_segments, N_FEATURES) feature matrix.
    list of int
        Segment IDs corresponding to each row.
    """
    segment_ids = sorted(np.unique(labels))
    n_segments = len(segment_ids)
    total_pixels = labels.size

    feature_matrix = np.zeros((n_segments, N_FEATURES), dtype=np.float64)

    for i, seg_id in enumerate(segment_ids):
        feature_matrix[i] = extract_segment_features(
            seg_id, labels, rgb, hsv, lab, gray, edges_canny,
            ndvi=ndvi, total_pixels=total_pixels,
        )
        if progress_callback is not None and i % 50 == 0:
            progress_callback(i, n_segments)

    if progress_callback is not None:
        progress_callback(n_segments, n_segments)

    return feature_matrix, segment_ids
