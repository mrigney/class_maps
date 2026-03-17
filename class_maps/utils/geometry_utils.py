"""Geometry utilities: segment shape metrics."""

import numpy as np


def compute_perimeter(mask):
    """Estimate perimeter of a binary mask by counting boundary pixels.

    Parameters
    ----------
    mask : np.ndarray
        (H, W) boolean mask.

    Returns
    -------
    int
        Estimated perimeter in pixels.
    """
    # Interior pixels: pixels where all 4-connected neighbors are also in the mask
    eroded = np.zeros_like(mask)
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    boundary = mask & ~eroded
    return int(boundary.sum())


def compute_compactness(area, perimeter):
    """Compute compactness (isoperimetric quotient).

    Compactness = 4 * pi * area / perimeter^2
    Circle = 1.0, elongated shapes < 1.0.

    Parameters
    ----------
    area : int
        Segment area in pixels.
    perimeter : int
        Segment perimeter in pixels.

    Returns
    -------
    float
        Compactness value in [0, 1].
    """
    if perimeter == 0:
        return 0.0
    return (4.0 * np.pi * area) / (perimeter ** 2)


def compute_eccentricity(mask):
    """Compute eccentricity of a segment from its second-order moments.

    Parameters
    ----------
    mask : np.ndarray
        (H, W) boolean mask.

    Returns
    -------
    float
        Eccentricity value. 0 = circle, approaching 1 = elongated.
    """
    rows, cols = np.where(mask)
    if len(rows) < 2:
        return 0.0

    # Center
    cr = rows.mean()
    cc = cols.mean()

    # Second-order central moments
    mu20 = ((rows - cr) ** 2).mean()
    mu02 = ((cols - cc) ** 2).mean()
    mu11 = ((rows - cr) * (cols - cc)).mean()

    # Eigenvalues of the covariance matrix
    diff = mu20 - mu02
    discriminant = np.sqrt(diff ** 2 + 4 * mu11 ** 2)
    lambda1 = (mu20 + mu02 + discriminant) / 2.0
    lambda2 = (mu20 + mu02 - discriminant) / 2.0

    if lambda1 <= 0:
        return 0.0

    return float(np.sqrt(1.0 - lambda2 / lambda1))
