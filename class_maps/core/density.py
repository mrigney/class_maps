"""Canopy density layer computation.

Produces a continuous 0.0-1.0 raster representing canopy/vegetation
cover fraction. Used by Houdini as a scatter density mask.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from class_maps.config import VEGETATION_CLASS_IDS


def compute_canopy_density(labels, segment_classes, gray, ndvi=None,
                           vegetation_class_ids=None):
    """Compute canopy density for vegetation superpixels.

    Density is estimated from a combination of:
    - Texture variance (GLCM energy inverse) — dense canopy has more texture
    - NDVI mean (if available) — higher NDVI = more vegetation
    - Green channel intensity relative to brightness

    Non-vegetation superpixels get density = 0.0.

    Parameters
    ----------
    labels : np.ndarray
        (H, W) int32 superpixel label array.
    segment_classes : dict
        {segment_id: class_id} from classification.
    gray : np.ndarray
        (H, W) uint8 grayscale image.
    ndvi : np.ndarray or None
        (H, W) float32 NDVI, or None.
    vegetation_class_ids : list of int, optional
        Class IDs considered vegetation. Default from config.

    Returns
    -------
    np.ndarray
        (H, W) float32 density raster in [0.0, 1.0].
    """
    if vegetation_class_ids is None:
        vegetation_class_ids = VEGETATION_CLASS_IDS

    h, w = labels.shape
    density = np.zeros((h, w), dtype=np.float32)

    # Collect density values for all vegetation segments
    veg_segments = {}  # {seg_id: raw_density}

    for seg_id, class_id in segment_classes.items():
        if class_id not in vegetation_class_ids:
            continue

        mask = labels == seg_id
        if mask.sum() < 4:
            continue

        # Texture component: standard deviation of grayscale within segment
        gray_vals = gray[mask].astype(np.float32)
        texture_score = np.std(gray_vals) / 128.0  # Normalize roughly to [0, 1]

        # NDVI component
        ndvi_score = 0.0
        if ndvi is not None:
            ndvi_vals = ndvi[mask]
            ndvi_mean = np.mean(ndvi_vals)
            # Map NDVI from [-1,1] to [0,1], with negative values = 0
            ndvi_score = np.clip(ndvi_mean, 0.0, 1.0)

        # Combine scores
        if ndvi is not None:
            raw_density = 0.4 * texture_score + 0.6 * ndvi_score
        else:
            # Without NDVI, rely more on texture
            raw_density = texture_score

        veg_segments[seg_id] = float(raw_density)

    if not veg_segments:
        return density

    # Normalize across all vegetation segments to [0, 1]
    values = np.array(list(veg_segments.values()))
    vmin, vmax = np.percentile(values, [5, 95])

    if vmax > vmin:
        for seg_id, raw in veg_segments.items():
            normalized = (raw - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0.0, 1.0)
            mask = labels == seg_id
            density[mask] = normalized
    else:
        # All segments have similar density
        for seg_id in veg_segments:
            mask = labels == seg_id
            density[mask] = 0.5

    return density
