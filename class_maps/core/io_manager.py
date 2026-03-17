"""Image I/O: load GeoTIFF and PNG, save classified/density rasters."""

import json
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image

from class_maps.config import GEOTIFF_EXTENSIONS, PNG_EXTENSIONS, SUPPORTED_EXTENSIONS

# Uniform container for loaded image data
ImageData = namedtuple("ImageData", ["array", "crs", "transform", "path", "has_nir"])


def load_image(path):
    """Load an image file, returning ImageData.

    Supports PNG (3-band RGB or RGBA) and GeoTIFF (3 or 4 band).
    For GeoTIFF, CRS and transform are preserved.
    For PNG, crs and transform will be None.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    ImageData
        Named tuple with array (H, W, C), crs, transform, path, has_nir.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}. "
                         f"Supported: {SUPPORTED_EXTENSIONS}")

    if ext in GEOTIFF_EXTENSIONS:
        return _load_geotiff(path)
    else:
        return _load_png(path)


def _load_png(path):
    """Load a PNG file as RGB numpy array."""
    img = Image.open(path)
    arr = np.array(img)

    # Handle RGBA -> RGB (drop alpha)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 2:
        # Grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)

    return ImageData(
        array=arr.astype(np.uint8),
        crs=None,
        transform=None,
        path=str(path),
        has_nir=False,
    )


def _load_geotiff(path):
    """Load a GeoTIFF file, preserving CRS and transform."""
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF support. "
            "Install it with: pip install rasterio"
        )

    with rasterio.open(path) as src:
        # Read all bands: rasterio returns (bands, H, W)
        data = src.read()
        crs = src.crs
        transform = src.transform

    # Transpose to (H, W, C)
    arr = np.transpose(data, (1, 2, 0))
    n_bands = arr.shape[2]

    has_nir = n_bands >= 4

    # Normalize to uint8 if needed (some GeoTIFFs are uint16 or float)
    if arr.dtype != np.uint8:
        if arr.dtype in (np.float32, np.float64):
            # Assume 0-1 range for floats
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                # Normalize to 0-255
                arr = _normalize_to_uint8(arr)
        elif arr.dtype == np.uint16:
            arr = _normalize_to_uint8(arr)
        else:
            arr = arr.astype(np.uint8)

    return ImageData(
        array=arr,
        crs=crs,
        transform=transform,
        path=str(path),
        has_nir=has_nir,
    )


def _normalize_to_uint8(arr):
    """Normalize array to 0-255 uint8 per-band."""
    result = np.zeros_like(arr, dtype=np.uint8)
    for i in range(arr.shape[2]):
        band = arr[:, :, i].astype(np.float64)
        bmin, bmax = np.percentile(band, [2, 98])
        if bmax > bmin:
            band = (band - bmin) / (bmax - bmin) * 255
        else:
            band = np.zeros_like(band)
        result[:, :, i] = band.clip(0, 255).astype(np.uint8)
    return result


def save_classified_raster(path, class_array, crs=None, transform=None):
    """Save a single-band classified raster as GeoTIFF.

    Parameters
    ----------
    path : str or Path
        Output file path.
    class_array : np.ndarray
        2D array (H, W) of class IDs (uint8).
    crs : rasterio CRS, optional
        Coordinate reference system.
    transform : rasterio Affine, optional
        Affine transform.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for GeoTIFF export.")

    h, w = class_array.shape
    kwargs = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint8",
    }
    if crs is not None:
        kwargs["crs"] = crs
    if transform is not None:
        kwargs["transform"] = transform

    with rasterio.open(path, "w", **kwargs) as dst:
        dst.write(class_array.astype(np.uint8), 1)


def save_density_raster(path, density_array, crs=None, transform=None):
    """Save a single-band density raster as GeoTIFF (float32, 0.0-1.0)."""
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for GeoTIFF export.")

    h, w = density_array.shape
    kwargs = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
    }
    if crs is not None:
        kwargs["crs"] = crs
    if transform is not None:
        kwargs["transform"] = transform

    with rasterio.open(path, "w", **kwargs) as dst:
        dst.write(density_array.astype(np.float32), 1)


def save_legend_json(path, class_definitions):
    """Save class legend as JSON.

    Parameters
    ----------
    path : str or Path
        Output JSON file path.
    class_definitions : dict
        {class_id: {"name": str, "color": (R, G, B)}}
    """
    # Convert tuples to lists for JSON serialization
    legend = {}
    for cid, cdef in class_definitions.items():
        legend[str(cid)] = {
            "name": cdef["name"],
            "color": list(cdef["color"]),
        }

    with open(path, "w") as f:
        json.dump(legend, f, indent=2)
