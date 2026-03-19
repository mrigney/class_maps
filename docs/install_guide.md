# class_maps Installation Guide

## System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 or later
- **RAM**: 4 GB minimum (8 GB recommended for images larger than 5000x5000 pixels)
- **Display**: 1024x768 minimum resolution

## Step 1: Clone or Download the Project

Place the `class_maps` project folder in your preferred working directory. The project structure should look like:

```
class_maps/
├── run.py
├── requirements.txt
├── class_maps/
│   ├── __init__.py
│   ├── config.py
│   ├── core/
│   ├── gui/
│   └── utils/
└── tests/
```

## Step 2: Create a Virtual Environment (Recommended)

Using a virtual environment keeps class_maps dependencies isolated from your system Python.

```bash
cd class_maps
python -m venv .venv
```

Activate the environment:

- **Windows (Command Prompt):** `.venv\Scripts\activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (Git Bash):** `source .venv/Scripts/activate`
- **Linux/macOS:** `source .venv/bin/activate`

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| numpy | Array computation |
| scikit-image | SLIC superpixels, GLCM texture, edge detection |
| scikit-learn | Random Forest classifier |
| opencv-python | Color space conversion, morphological operations |
| Pillow | PNG image loading |
| PyQt5 | GUI framework |
| rasterio | GeoTIFF I/O with CRS preservation |

**Optional (for U-Net road detection):**

| Package | Purpose |
|---------|---------|
| torch | PyTorch deep learning framework |
| torchvision | Image transforms and pretrained models |
| segmentation-models-pytorch | U-Net architecture with pretrained encoders |

### Troubleshooting: rasterio on Windows

If `pip install rasterio` fails, rasterio requires the GDAL library. Try one of these approaches:

1. **Conda** (easiest on Windows):
   ```bash
   conda install -c conda-forge rasterio
   ```

2. **Pre-built wheel**: Download a compatible `.whl` file from [Christoph Gohlke's archive](https://github.com/cgohlke/geospatial-wheels/) and install with:
   ```bash
   pip install rasterio-<version>-<platform>.whl
   ```

3. **Without rasterio**: class_maps will still work with PNG files. GeoTIFF support requires rasterio. You will see a clear error message if you try to open a GeoTIFF without rasterio installed.

### Troubleshooting: PyQt5

On some Linux distributions, PyQt5 may require system-level dependencies:

```bash
# Ubuntu/Debian
sudo apt install python3-pyqt5

# Fedora
sudo dnf install python3-qt5
```

## Step 4: Verify Installation

Run a quick check that all imports work:

```bash
python -c "from class_maps.gui.main_window import ClassMapsWindow; print('OK')"
```

If this prints `OK` without errors, the installation is complete.

## Step 5: Launch the Application

```bash
python run.py
```

The class_maps window should appear. See the User's Guide for usage instructions.

## Optional: Train the Road Segmentation Model

class_maps includes an optional U-Net model for automatic road detection. This requires PyTorch and a GPU is strongly recommended.

### Install PyTorch dependencies

```bash
pip install torch torchvision segmentation-models-pytorch
```

See [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific install commands (CUDA version selection, etc.).

### Train the model

```bash
python -m class_maps.train_road_model
```

This will:
1. Download the Massachusetts Roads Dataset (~5.8 GB) from the University of Toronto
2. Train a U-Net (ResNet34 encoder) for 25 epochs
3. Save weights to `~/.class_maps/models/road_unet_resnet34.pth`

Training time: ~15 minutes with a modern GPU, ~1+ hours on CPU.

If the download fails or times out, re-run the same command — it resumes from where it left off. Alternatively, download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset) and use:

```bash
python -m class_maps.train_road_model --data-dir /path/to/mass_roads --skip-download
```

Once trained, the model is automatically used whenever "Detect linear features" is enabled in the GUI. No further configuration is needed.

### Skipping the model

The U-Net is entirely optional. Without it, class_maps falls back to heuristic linear feature detection, or you can draw roads manually using the Draw tool. The core classification workflow (superpixels + Random Forest) works without PyTorch.

## Updating

To update class_maps after pulling new code:

```bash
pip install -r requirements.txt --upgrade
```

## Uninstalling

Deactivate and remove the virtual environment:

```bash
deactivate
rm -rf .venv     # Linux/macOS
rmdir /s .venv   # Windows
```
