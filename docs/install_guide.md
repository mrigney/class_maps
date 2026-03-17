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
