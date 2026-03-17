"""Image preprocessing: color space conversion, NDVI, edge detection."""

import cv2
import numpy as np


def to_hsv(rgb_array):
    """Convert RGB uint8 array to HSV.

    Parameters
    ----------
    rgb_array : np.ndarray
        (H, W, 3) uint8 RGB image.

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 HSV image.
    """
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)


def to_lab(rgb_array):
    """Convert RGB uint8 array to CIE Lab.

    Parameters
    ----------
    rgb_array : np.ndarray
        (H, W, 3) uint8 RGB image.

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 Lab image.
    """
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2Lab)


def to_grayscale(rgb_array):
    """Convert RGB uint8 array to grayscale.

    Returns
    -------
    np.ndarray
        (H, W) uint8 grayscale image.
    """
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)


def compute_ndvi(nir_band, red_band):
    """Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Parameters
    ----------
    nir_band : np.ndarray
        (H, W) NIR band (uint8 or float).
    red_band : np.ndarray
        (H, W) Red band (uint8 or float).

    Returns
    -------
    np.ndarray
        (H, W) float32 NDVI in range [-1, 1].
    """
    nir = nir_band.astype(np.float32)
    red = red_band.astype(np.float32)
    denominator = nir + red
    # Avoid division by zero
    ndvi = np.where(denominator > 0, (nir - red) / denominator, 0.0)
    return ndvi.astype(np.float32)


def detect_edges_canny(gray, low_threshold=50, high_threshold=150):
    """Detect edges using Canny edge detector.

    Parameters
    ----------
    gray : np.ndarray
        (H, W) uint8 grayscale image.
    low_threshold : int
        Lower threshold for hysteresis.
    high_threshold : int
        Upper threshold for hysteresis.

    Returns
    -------
    np.ndarray
        (H, W) uint8 binary edge map (0 or 255).
    """
    return cv2.Canny(gray, low_threshold, high_threshold)


def detect_edges_sobel(gray):
    """Detect edges using Sobel operator (gradient magnitude).

    Parameters
    ----------
    gray : np.ndarray
        (H, W) uint8 grayscale image.

    Returns
    -------
    np.ndarray
        (H, W) float32 gradient magnitude.
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return magnitude.astype(np.float32)


def preprocess_image(image_array, has_nir=False):
    """Run all preprocessing steps on an image.

    Parameters
    ----------
    image_array : np.ndarray
        (H, W, C) uint8 image. C=3 for RGB, C=4 for RGB+NIR.
    has_nir : bool
        Whether the image has a 4th NIR band.

    Returns
    -------
    dict
        Keys: 'rgb', 'hsv', 'lab', 'gray', 'ndvi' (or None), 'edges_canny', 'edges_sobel'.
    """
    rgb = image_array[:, :, :3]
    hsv = to_hsv(rgb)
    lab = to_lab(rgb)
    gray = to_grayscale(rgb)
    edges_canny = detect_edges_canny(gray)
    edges_sobel = detect_edges_sobel(gray)

    ndvi = None
    if has_nir and image_array.shape[2] >= 4:
        nir_band = image_array[:, :, 3]
        red_band = rgb[:, :, 0]
        ndvi = compute_ndvi(nir_band, red_band)

    return {
        "rgb": rgb,
        "hsv": hsv,
        "lab": lab,
        "gray": gray,
        "ndvi": ndvi,
        "edges_canny": edges_canny,
        "edges_sobel": edges_sobel,
    }
