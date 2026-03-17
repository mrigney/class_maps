"""U-Net road segmentation model: inference, weight loading, tile-based prediction.

Uses segmentation_models_pytorch for the U-Net architecture with an ImageNet-
pretrained ResNet34 encoder. Trained weights are stored in the user's data
directory and loaded automatically when available.

Tile-based inference handles images of any size by splitting into overlapping
tiles, predicting each, and stitching with blending in overlap regions.
"""

import os
import numpy as np

# Model weight search paths (checked in order)
_WEIGHT_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "models"),  # project/models/
    os.path.join(os.path.expanduser("~"), ".class_maps", "models"),  # ~/.class_maps/models/
]
WEIGHT_FILENAME = "road_unet_resnet34.pth"

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Inference tile size (must match training)
TILE_SIZE = 512
TILE_OVERLAP = 64


def get_weight_path():
    """Find trained model weights, searching known locations.

    Returns
    -------
    str or None
        Path to the .pth weight file, or None if not found.
    """
    for d in _WEIGHT_DIRS:
        path = os.path.join(d, WEIGHT_FILENAME)
        if os.path.isfile(path):
            return path
    return None


def get_default_weight_dir():
    """Return the default directory for saving trained weights."""
    d = os.path.join(os.path.expanduser("~"), ".class_maps", "models")
    os.makedirs(d, exist_ok=True)
    return d


def create_model():
    """Create a U-Net model with ResNet34 encoder.

    Returns
    -------
    torch.nn.Module
        The U-Net model (on CPU, eval mode if weights loaded).
    """
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model


def load_model(weight_path=None):
    """Load a trained road segmentation model.

    Parameters
    ----------
    weight_path : str, optional
        Explicit path to .pth file. If None, searches default locations.

    Returns
    -------
    torch.nn.Module or None
        Loaded model in eval mode, or None if no weights found.
    """
    import torch

    if weight_path is None:
        weight_path = get_weight_path()

    if weight_path is None:
        return None

    model = create_model()
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_mask(model, rgb, confidence=0.5, progress_callback=None):
    """Run road segmentation on a full image using tile-based inference.

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net model.
    rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    confidence : float
        Probability threshold for road classification (0.0-1.0).
    progress_callback : callable, optional
        Called with (current_tile, total_tiles) for progress reporting.

    Returns
    -------
    np.ndarray
        (H, W) bool mask of detected road pixels.
    """
    import torch

    h, w = rgb.shape[:2]
    device = _get_device()
    model = model.to(device)
    model.eval()

    # Accumulator for probability map (with blending weights)
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    # Generate tile positions
    tiles = _generate_tiles(h, w, TILE_SIZE, TILE_OVERLAP)
    total_tiles = len(tiles)

    # Create blending weight mask (cosine falloff at edges)
    blend_weight = _create_blend_weight(TILE_SIZE, TILE_OVERLAP)

    with torch.no_grad():
        for i, (y0, x0, y1, x1) in enumerate(tiles):
            tile = rgb[y0:y1, x0:x1]

            # Pad if tile is smaller than TILE_SIZE
            th, tw = tile.shape[:2]
            if th < TILE_SIZE or tw < TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            # Normalize and convert to tensor
            tile_f = tile.astype(np.float32) / 255.0
            tile_f = (tile_f - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(tile_f.transpose(2, 0, 1)).unsqueeze(0)
            tensor = tensor.to(device)

            # Predict
            logits = model(tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Crop padding if needed
            prob = prob[:th, :tw]
            bw = blend_weight[:th, :tw]

            # Accumulate with blending
            prob_sum[y0:y1, x0:x1] += prob * bw
            weight_sum[y0:y1, x0:x1] += bw

            if progress_callback:
                progress_callback(i + 1, total_tiles)

    # Normalize by weights
    weight_sum = np.maximum(weight_sum, 1e-6)
    prob_map = prob_sum / weight_sum

    return prob_map >= confidence


def predict_probability(model, rgb, progress_callback=None):
    """Run road segmentation and return the probability map (no thresholding).

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net model.
    rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    progress_callback : callable, optional
        Called with (current_tile, total_tiles).

    Returns
    -------
    np.ndarray
        (H, W) float32 probability map (0.0-1.0).
    """
    import torch

    h, w = rgb.shape[:2]
    device = _get_device()
    model = model.to(device)
    model.eval()

    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    tiles = _generate_tiles(h, w, TILE_SIZE, TILE_OVERLAP)
    blend_weight = _create_blend_weight(TILE_SIZE, TILE_OVERLAP)

    with torch.no_grad():
        for i, (y0, x0, y1, x1) in enumerate(tiles):
            tile = rgb[y0:y1, x0:x1]
            th, tw = tile.shape[:2]
            if th < TILE_SIZE or tw < TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            tile_f = tile.astype(np.float32) / 255.0
            tile_f = (tile_f - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(tile_f.transpose(2, 0, 1)).unsqueeze(0)
            tensor = tensor.to(device)

            logits = model(tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            prob = prob[:th, :tw]
            bw = blend_weight[:th, :tw]

            prob_sum[y0:y1, x0:x1] += prob * bw
            weight_sum[y0:y1, x0:x1] += bw

            if progress_callback:
                progress_callback(i + 1, total_tiles)

    weight_sum = np.maximum(weight_sum, 1e-6)
    return prob_sum / weight_sum


def _get_device():
    """Select the best available device (CUDA > CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _generate_tiles(h, w, tile_size, overlap):
    """Generate (y0, x0, y1, x1) tile coordinates with overlap.

    Returns
    -------
    list of tuple
        Each tuple is (y0, x0, y1, x1) defining a tile region.
    """
    tiles = []
    step = tile_size - overlap

    y = 0
    while y < h:
        y1 = min(y + tile_size, h)
        x = 0
        while x < w:
            x1 = min(x + tile_size, w)
            tiles.append((y, x, y1, x1))
            if x1 >= w:
                break
            x += step
        if y1 >= h:
            break
        y += step

    return tiles


def _create_blend_weight(tile_size, overlap):
    """Create a 2D blending weight with cosine ramps at edges.

    Center pixels get weight 1.0, edges ramp down over the overlap region.

    Returns
    -------
    np.ndarray
        (tile_size, tile_size) float32 weight array.
    """
    ramp = np.ones(tile_size, dtype=np.float32)
    if overlap > 0:
        # Ramp up at start, ramp down at end
        for i in range(overlap):
            w = 0.5 * (1 - np.cos(np.pi * i / overlap))
            ramp[i] = w
            ramp[tile_size - 1 - i] = w

    # 2D weight = outer product
    weight = np.outer(ramp, ramp)
    return weight
