"""Train a U-Net road segmentation model on the Massachusetts Roads Dataset.

Usage:
    python -m class_maps.train_road_model [--data-dir DIR] [--epochs N] [--batch-size N]

This script:
1. Downloads the Massachusetts Roads Dataset (if not already present)
2. Trains a U-Net (ResNet34 encoder, ImageNet pretrained) for binary road segmentation
3. Saves trained weights to ~/.class_maps/models/road_unet_resnet34.pth

The trained model is automatically used by the linear feature detection module.
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np

# ImageNet normalization constants (shared with road_model.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training defaults
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-4
DEFAULT_PATCH_SIZE = 512
PATCHES_PER_IMAGE = 4  # random crops per 1500x1500 image per epoch


def get_default_data_dir():
    """Return default directory for dataset storage."""
    return os.path.join(os.path.expanduser("~"), ".class_maps", "data", "mass_roads")


def download_dataset(data_dir):
    """Download the Massachusetts Roads Dataset from the original source.

    Falls back to instructions for Kaggle download if the original server
    is unavailable.

    Parameters
    ----------
    data_dir : str
        Directory to save the dataset.
    """
    import urllib.request
    import urllib.error

    base_url = "https://www.cs.toronto.edu/~vmnih/data/mass_roads"
    splits = ["train", "valid", "test"]
    types = ["sat", "map"]

    os.makedirs(data_dir, exist_ok=True)

    # Check if already downloaded
    train_sat = os.path.join(data_dir, "train", "sat")
    if os.path.isdir(train_sat) and len(os.listdir(train_sat)) > 100:
        print(f"Dataset already present at {data_dir} ({len(os.listdir(train_sat))} training images)")
        return True

    print("Downloading Massachusetts Roads Dataset...")
    print(f"Target directory: {data_dir}")

    # Try to get index page to check availability
    try:
        index_url = f"{base_url}/train/sat/"
        req = urllib.request.Request(index_url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
    except (urllib.error.URLError, OSError):
        print("\nOriginal dataset server is unavailable.")
        print("Please download manually from Kaggle:")
        print("  1. Visit: https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset")
        print("  2. Download and extract to:", data_dir)
        print("  3. Expected structure:")
        print("     mass_roads/train/sat/*.tiff")
        print("     mass_roads/train/map/*.tiff")
        print("     mass_roads/valid/sat/*.tiff")
        print("     mass_roads/valid/map/*.tiff")
        print("\nOr use the Kaggle CLI:")
        print(f"  kaggle datasets download -d balraj98/massachusetts-roads-dataset -p {data_dir} --unzip")
        return False

    # Download files by scraping the directory listing
    for split in splits:
        for dtype in types:
            split_dir = os.path.join(data_dir, split, dtype)
            os.makedirs(split_dir, exist_ok=True)

            listing_url = f"{base_url}/{split}/{dtype}/"
            print(f"Fetching file list: {split}/{dtype}/...")

            try:
                with urllib.request.urlopen(listing_url, timeout=30) as resp:
                    html = resp.read().decode("utf-8")

                # Parse filenames from directory listing
                import re
                filenames = re.findall(r'href="([^"]+\.tiff?)"', html, re.IGNORECASE)
                if not filenames:
                    filenames = re.findall(r'href="([^"]+\.png)"', html, re.IGNORECASE)

                print(f"  Found {len(filenames)} files")

                for i, fname in enumerate(filenames):
                    out_path = os.path.join(split_dir, fname)
                    if os.path.exists(out_path):
                        continue

                    file_url = f"{listing_url}{fname}"
                    print(f"  [{i+1}/{len(filenames)}] {fname}", end="\r")
                    urllib.request.urlretrieve(file_url, out_path)

                print(f"  {split}/{dtype}: {len(filenames)} files done")

            except (urllib.error.URLError, OSError) as e:
                print(f"  Error downloading {split}/{dtype}: {e}")
                return False

    print("Download complete!")
    return True


class RoadDataset:
    """PyTorch-compatible dataset for Massachusetts Roads patches."""

    def __init__(self, data_dir, split="train", patch_size=512, augment=True):
        self.patch_size = patch_size
        self.augment = augment

        sat_dir = os.path.join(data_dir, split, "sat")
        map_dir = os.path.join(data_dir, split, "map")

        # Find matching image/mask pairs
        self.pairs = []
        if os.path.isdir(sat_dir):
            for fname in sorted(os.listdir(sat_dir)):
                sat_path = os.path.join(sat_dir, fname)
                # Try both same extension and common alternatives
                stem = Path(fname).stem
                map_path = None
                for ext in [Path(fname).suffix, ".tiff", ".tif", ".png"]:
                    candidate = os.path.join(map_dir, stem + ext)
                    if os.path.exists(candidate):
                        map_path = candidate
                        break
                if map_path:
                    self.pairs.append((sat_path, map_path))

        if not self.pairs:
            raise FileNotFoundError(
                f"No image/mask pairs found in {data_dir}/{split}/. "
                f"Expected sat/ and map/ subdirectories with matching filenames."
            )

    def __len__(self):
        return len(self.pairs) * PATCHES_PER_IMAGE

    def __getitem__(self, idx):
        import torch
        from PIL import Image

        pair_idx = idx // PATCHES_PER_IMAGE
        sat_path, map_path = self.pairs[pair_idx]

        # Load image and mask
        img = np.array(Image.open(sat_path).convert("RGB"))
        mask = np.array(Image.open(map_path).convert("L"))

        # Random crop
        h, w = img.shape[:2]
        ps = self.patch_size
        if h > ps and w > ps:
            y = random.randint(0, h - ps)
            x = random.randint(0, w - ps)
            img = img[y:y+ps, x:x+ps]
            mask = mask[y:y+ps, x:x+ps]
        else:
            # Resize if image is smaller than patch size
            img = np.array(Image.fromarray(img).resize((ps, ps)))
            mask = np.array(Image.fromarray(mask).resize((ps, ps), Image.NEAREST))

        # Augmentation
        if self.augment:
            # Random flip
            if random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if random.random() > 0.5:
                img = np.flip(img, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            # Random 90-degree rotation
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()

        # Normalize image
        img_f = img.astype(np.float32) / 255.0
        for c in range(3):
            img_f[:, :, c] = (img_f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        # Binarize mask (255 = road → 1.0)
        mask_f = (mask > 128).astype(np.float32)

        # Convert to tensors (C, H, W)
        img_tensor = torch.from_numpy(img_f.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask_f).unsqueeze(0)

        return img_tensor, mask_tensor


def train(data_dir, epochs, batch_size, lr, patch_size):
    """Train the U-Net road segmentation model.

    Parameters
    ----------
    data_dir : str
        Path to the Massachusetts Roads dataset.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    patch_size : int
        Size of random crops for training.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from class_maps.core.road_model import create_model, get_default_weight_dir

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU will be slow. GPU is recommended.")
        print("         Expected time: ~1 hour on CPU, ~15 min on GPU.")

    # Create datasets
    print(f"\nLoading dataset from {data_dir}...")
    train_dataset = RoadDataset(data_dir, split="train", patch_size=patch_size)
    print(f"Training samples: {len(train_dataset)} patches "
          f"({len(train_dataset.pairs)} images x {PATCHES_PER_IMAGE} patches)")

    try:
        val_dataset = RoadDataset(data_dir, split="valid", patch_size=patch_size,
                                  augment=False)
        print(f"Validation samples: {len(val_dataset)} patches")
        has_val = True
    except FileNotFoundError:
        print("No validation set found, skipping validation.")
        has_val = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    if has_val:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=(device.type == "cuda"))

    # Create model
    model = create_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss: BCE + Dice for handling class imbalance
    bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        smooth = 1.0
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def combined_loss(pred, target):
        return bce_loss(pred, target) + dice_loss(pred, target)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Training loop
    best_val_loss = float("inf")
    weight_dir = get_default_weight_dir()
    weight_path = os.path.join(weight_dir, "road_unet_resnet34.pth")

    print(f"\nTraining for {epochs} epochs...")
    print(f"Weights will be saved to: {weight_path}\n")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {train_loss/n_batches:.4f}")

        avg_train_loss = train_loss / max(n_batches, 1)

        # --- Validate ---
        if has_val:
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            n_val = 0

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    outputs = model(images)
                    loss = combined_loss(outputs, masks)
                    val_loss += loss.item()

                    # Compute IoU
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    intersection = (preds * masks).sum()
                    union = preds.sum() + masks.sum() - intersection
                    iou = (intersection + 1e-6) / (union + 1e-6)
                    val_iou += iou.item()
                    n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)
            avg_val_iou = val_iou / max(n_val, 1)

            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

            scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), weight_path)
                print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")
        else:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f}")
            # Save every epoch when no validation set
            torch.save(model.state_dict(), weight_path)

    # Save final model if no validation set
    if not has_val:
        torch.save(model.state_dict(), weight_path)

    print(f"\nTraining complete!")
    print(f"Model saved to: {weight_path}")
    print(f"The model will be automatically used for linear feature detection.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a U-Net road segmentation model on Massachusetts Roads Dataset"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"Dataset directory (default: {get_default_data_dir()})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
                        help=f"Training patch size (default: {DEFAULT_PATCH_SIZE})")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (assume already present)")
    args = parser.parse_args()

    data_dir = args.data_dir or get_default_data_dir()

    # Step 1: Download dataset
    if not args.skip_download:
        success = download_dataset(data_dir)
        if not success:
            print("\nDataset download failed. After manually downloading, run:")
            print(f"  python -m class_maps.train_road_model --data-dir {data_dir} --skip-download")
            sys.exit(1)

    # Step 2: Train
    train(data_dir, args.epochs, args.batch_size, args.lr, args.patch_size)


if __name__ == "__main__":
    main()
