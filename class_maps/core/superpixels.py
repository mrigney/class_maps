"""SLIC superpixel segmentation and segment property computation."""

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


def get_segment_mask(labels, segment_id):
    """Get a boolean mask for a specific segment.

    Parameters
    ----------
    labels : np.ndarray
        (H, W) segment label array.
    segment_id : int
        The segment ID to extract.

    Returns
    -------
    np.ndarray
        (H, W) boolean mask.
    """
    return labels == segment_id


def get_segment_ids(labels):
    """Get sorted list of unique segment IDs.

    Parameters
    ----------
    labels : np.ndarray
        (H, W) segment label array.

    Returns
    -------
    np.ndarray
        Sorted array of unique segment IDs.
    """
    return np.unique(labels)


def get_boundary_image(rgb_array, labels, color=(255, 255, 0)):
    """Generate an image with superpixel boundaries drawn.

    Parameters
    ----------
    rgb_array : np.ndarray
        (H, W, 3) uint8 RGB image.
    labels : np.ndarray
        (H, W) segment label array.
    color : tuple
        RGB color for boundary lines.

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 image with boundaries drawn.
    """
    # mark_boundaries returns float [0, 1]
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

    Parameters
    ----------
    labels : np.ndarray
        (H, W) segment label array.

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
