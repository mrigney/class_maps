"""Post-processing: morphological cleanup, shadow resolution, boundary refinement."""

import numpy as np
from scipy import ndimage

from class_maps.config import SHADOW_V_THRESHOLD, SHADOW_S_THRESHOLD, VEGETATION_CLASS_IDS


def labels_to_pixel_raster(labels, segment_classes):
    """Convert superpixel-level classifications to a full-resolution class raster.

    Parameters
    ----------
    labels : np.ndarray
        (H, W) int32 superpixel label array.
    segment_classes : dict
        {segment_id: class_id}

    Returns
    -------
    np.ndarray
        (H, W) uint8 class raster.
    """
    h, w = labels.shape
    raster = np.zeros((h, w), dtype=np.uint8)
    for seg_id, class_id in segment_classes.items():
        raster[labels == seg_id] = class_id
    return raster


def morphological_cleanup(class_raster, labels, min_region_pixels=50):
    """Remove tiny classified regions by majority vote from neighbors.

    Small isolated regions (fewer than min_region_pixels) are reassigned
    to the most common neighboring class.

    Parameters
    ----------
    class_raster : np.ndarray
        (H, W) uint8 class raster.
    labels : np.ndarray
        (H, W) int32 superpixel label array.
    min_region_pixels : int
        Minimum region size. Smaller regions get merged.

    Returns
    -------
    np.ndarray
        (H, W) uint8 cleaned class raster.
    """
    result = class_raster.copy()
    unique_classes = np.unique(result)

    for class_id in unique_classes:
        if class_id == 0:
            continue

        class_mask = result == class_id
        labeled_regions, n_regions = ndimage.label(class_mask)

        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            region_size = region_mask.sum()

            if region_size >= min_region_pixels:
                continue

            # Find neighboring classes using dilation
            dilated = ndimage.binary_dilation(region_mask, iterations=2)
            border = dilated & ~region_mask

            if not border.any():
                continue

            neighbor_classes = result[border]
            neighbor_classes = neighbor_classes[neighbor_classes != class_id]
            neighbor_classes = neighbor_classes[neighbor_classes != 0]

            if len(neighbor_classes) == 0:
                continue

            # Majority vote
            values, counts = np.unique(neighbor_classes, return_counts=True)
            replacement = values[np.argmax(counts)]
            result[region_mask] = replacement

    return result


def detect_shadow_segments(labels, hsv, segment_ids):
    """Detect superpixels that are likely shadows.

    Parameters
    ----------
    labels : np.ndarray
        (H, W) int32 superpixel label array.
    hsv : np.ndarray
        (H, W, 3) uint8 HSV image.
    segment_ids : list of int
        All segment IDs.

    Returns
    -------
    set of int
        Segment IDs flagged as shadow.
    """
    shadow_segments = set()

    for seg_id in segment_ids:
        mask = labels == seg_id
        v_mean = hsv[mask, 2].mean()
        s_mean = hsv[mask, 1].mean()

        # Shadows: low brightness, low-to-moderate saturation
        if v_mean < SHADOW_V_THRESHOLD and s_mean < SHADOW_S_THRESHOLD:
            shadow_segments.add(seg_id)

    return shadow_segments


def resolve_shadows(class_raster, labels, segment_ids, shadow_segments,
                    feature_matrix, seg_id_to_row):
    """Reassign shadow segments to the most similar neighboring class.

    Parameters
    ----------
    class_raster : np.ndarray
        (H, W) uint8 class raster.
    labels : np.ndarray
        (H, W) int32 superpixel label array.
    segment_ids : list of int
        All segment IDs.
    shadow_segments : set of int
        Segment IDs flagged as shadow.
    feature_matrix : np.ndarray
        (n_segments, n_features) feature matrix.
    seg_id_to_row : dict
        {segment_id: row_index} mapping.

    Returns
    -------
    np.ndarray
        (H, W) uint8 class raster with shadows resolved.
    """
    if not shadow_segments:
        return class_raster

    result = class_raster.copy()

    for seg_id in shadow_segments:
        if seg_id not in seg_id_to_row:
            continue

        mask = labels == seg_id
        # Find neighboring segments
        dilated = ndimage.binary_dilation(mask, iterations=3)
        border = dilated & ~mask

        neighbor_seg_ids = set(labels[border].flat) - {seg_id} - shadow_segments
        if not neighbor_seg_ids:
            continue

        # Find the most similar neighbor by feature distance
        shadow_features = feature_matrix[seg_id_to_row[seg_id]]
        best_dist = float("inf")
        best_class = None

        for n_id in neighbor_seg_ids:
            if n_id not in seg_id_to_row:
                continue
            n_features = feature_matrix[seg_id_to_row[n_id]]
            dist = np.linalg.norm(shadow_features - n_features)
            if dist < best_dist:
                best_dist = dist
                best_class = result[labels == n_id][0]

        if best_class is not None and best_class != 0:
            result[mask] = best_class

    return result


def refine_boundaries(class_raster, edges_canny, iterations=1):
    """Snap classification boundaries to detected edges.

    Where classification boundaries nearly align with strong edges
    (within a few pixels), shift the boundary to the edge.

    Parameters
    ----------
    class_raster : np.ndarray
        (H, W) uint8 class raster.
    edges_canny : np.ndarray
        (H, W) uint8 binary edge map.
    iterations : int
        Number of refinement passes.

    Returns
    -------
    np.ndarray
        (H, W) uint8 refined class raster.
    """
    result = class_raster.copy()
    edge_mask = edges_canny > 0

    for _ in range(iterations):
        # Find classification boundary pixels
        class_boundary = np.zeros_like(result, dtype=bool)
        class_boundary[:, :-1] |= result[:, :-1] != result[:, 1:]
        class_boundary[:, 1:] |= result[:, :-1] != result[:, 1:]
        class_boundary[:-1, :] |= result[:-1, :] != result[1:, :]
        class_boundary[1:, :] |= result[:-1, :] != result[1:, :]

        # Pixels near edges that are also near class boundaries
        edge_dilated = ndimage.binary_dilation(edge_mask, iterations=2)
        boundary_dilated = ndimage.binary_dilation(class_boundary, iterations=2)

        # In the overlap zone, use local majority to clean up
        refinement_zone = edge_dilated & boundary_dilated
        if not refinement_zone.any():
            break

        # For each pixel in the refinement zone, assign to local majority
        for class_id in np.unique(result):
            if class_id == 0:
                continue
            class_mask = result == class_id
            # Count how many neighbors of each pixel belong to this class
            neighbor_count = ndimage.uniform_filter(
                class_mask.astype(np.float32), size=5
            )
            # If this class has majority in a 5x5 window and pixel is in
            # refinement zone, assign it
            assign = refinement_zone & (neighbor_count > 0.5) & ~edge_mask
            result[assign] = class_id

    return result
