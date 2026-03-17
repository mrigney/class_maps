# class_maps User's Guide

## Overview

class_maps is a semi-supervised landcover classification tool for satellite imagery. You provide a satellite image, click a few example pixels for each landcover class, and the tool classifies the entire image. The output is a labeled raster suitable for use as Houdini heightfield masks, plus a canopy density raster for driving vegetation scatter density.

## Launching

```bash
python run.py
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open image |
| Ctrl+E | Export results |
| Ctrl+S | Save terrain profile |
| Ctrl+L | Load terrain profile |
| Ctrl+0 | Fit image to window |
| Ctrl+Q | Quit |

## Basic Workflow

### 1. Open an Image

**File > Open Image** (Ctrl+O) and select a satellite image. Supported formats:

- **PNG** (3-band RGB or RGBA; alpha is discarded)
- **GeoTIFF** (3-band RGB or 4-band RGB+NIR)

For GeoTIFF files, the CRS and affine transform are preserved and written to all output files.

Once loaded, the tool automatically computes SLIC superpixels. The status bar shows progress and reports the number of segments when complete.

### 2. Adjust Superpixels (Optional)

The right sidebar contains superpixel controls:

- **Segments**: Number of superpixels (default: 2000). Higher values produce smaller, more detailed segments. Lower values produce larger, coarser segments.
- **Compactness**: Balance between color similarity and spatial proximity (default: 10.0). Higher values produce more regularly shaped segments. Lower values produce segments that follow color boundaries more closely.

Toggle **Superpixel Boundaries** in the Overlays section to visualize segment edges. This is helpful for understanding the segmentation before labeling.

Click **Recompute Superpixels** after changing parameters. Note: this clears all existing labels.

**Guidelines for choosing superpixel count:**

| Image Size | Recommended Segments |
|------------|---------------------|
| ~1500x1500 | 500-1000 |
| ~3000x3000 | 1000-2000 |
| ~5000x5000 | 2000-4000 |

The goal is for each superpixel to contain a single landcover type. If superpixels are too large, they will span multiple cover types and classification accuracy suffers. If they are too small, feature extraction becomes less meaningful and computation is slower.

### 3. Label Training Samples

This is the most important step. The quality of your classification depends directly on the quality and quantity of your training labels.

1. **Select a class** in the "Landcover Classes" panel on the right sidebar. Click on the class name to make it active. The active class is shown below the list.

2. **Left-click on the image** to label the superpixel under your cursor as the active class. The labeled superpixel is highlighted with the class color.

3. **Right-click** on a labeled superpixel to remove its label.

4. Repeat for all classes you want to identify in the image.

**Guidelines for good training labels:**

- Label **at least 3-5 superpixels per class**, ideally more. The Random Forest needs enough examples to learn the variation within each class.
- Label examples that cover the **range of appearances** within each class. For example, if some grass is sunlit and some is in shadow, label examples of both.
- Label examples from **different parts of the image**. Don't cluster all labels in one corner.
- You don't need to label every class in the default palette. Only label classes that are actually present in the image.
- It's better to label **a few confident examples** than many uncertain ones. If you're not sure whether a superpixel is grass or agriculture, skip it.
- The status bar shows the segment ID and RGB values under your cursor, which can help with uncertain areas.

### 4. Classify

Click the **Classify** button (enabled once at least 2 classes have labels). The pipeline runs:

1. **Feature extraction** (first time only, cached afterward): computes a 30-dimensional feature vector for every superpixel, including color statistics in three color spaces, texture features, edge density, and shape metrics. This may take a few seconds.

2. **Random Forest training**: trains on your labeled superpixels. The status bar reports the Out-of-Bag (OOB) accuracy, which estimates how well the classifier generalizes. Aim for >80% OOB accuracy.

3. **Prediction**: classifies all unlabeled superpixels.

4. **Post-processing**:
   - Removes tiny isolated regions by merging them into neighboring classes
   - Automatically detects shadow regions and reassigns them to the most plausible underlying class
   - Snaps classification boundaries to detected image edges

5. **Density computation**: generates a canopy density layer for vegetation classes.

The classification overlay appears automatically. Use the **Overlays** controls to toggle it on/off and adjust opacity.

### 5. Iterate

Classification is rarely perfect on the first pass. The iterative workflow is:

1. Inspect the classification overlay. Zoom in to areas of interest.
2. Find misclassified superpixels.
3. Label them with the correct class (left-click).
4. Click **Classify** again. The feature matrix is cached, so reclassification is fast.
5. Repeat until satisfied.

Each iteration typically improves results significantly. Adding 5-10 corrective labels and reclassifying is usually enough to fix problem areas.

### 6. Review the Density Layer

Toggle **Canopy Density** in the Overlays section. This shows a green heatmap where brightness indicates vegetation density:

- Bright green = dense canopy (e.g., closed-canopy forest)
- Dim green = sparse vegetation (e.g., scattered trees in pasture)
- No overlay = non-vegetation areas (roads, water, bare soil)

The density layer only applies to vegetation classes (Coniferous forest, Deciduous forest, Shrub/scrub by default). When you bring this into Houdini, use it as a scatter density mask for tree/vegetation placement.

### 7. Export Results

**File > Export Results** (Ctrl+E) opens the export dialog:

- **Output directory**: where to save files
- **Filename prefix**: base name for output files (default: "classified")
- **Export options**:
  - Classified raster (`<prefix>_classes.tif`): single-band uint8 GeoTIFF where each pixel value is a class ID
  - Density raster (`<prefix>_density.tif`): single-band float32 GeoTIFF with values 0.0-1.0
  - Class legend (`<prefix>_legend.json`): JSON mapping class IDs to names and colors

If the source image was a GeoTIFF, the CRS and transform are preserved in the output files.

## Managing Classes

### Default Classes

| ID | Class | Typical Use |
|----|-------|-------------|
| 1 | Coniferous forest | Pine, spruce, fir stands |
| 2 | Deciduous forest | Broadleaf tree stands |
| 3 | Shrub/scrub | Low woody vegetation |
| 4 | Grass/field | Meadows, pastures, lawns |
| 5 | Agriculture | Cultivated cropland |
| 6 | Bare soil | Exposed earth, sand |
| 7 | Road (paved) | Asphalt, concrete surfaces |
| 8 | Road (unpaved) | Gravel, dirt roads and paths |
| 9 | Water | Lakes, ponds, rivers, streams |
| 10 | Structure/building | Man-made structures |

### Adding a Class

Click **Add** in the Landcover Classes panel. Enter a name and the class is created with an automatically generated color.

### Removing a Class

Select a class and click **Remove**. Any superpixels labeled with that class will have their labels cleared.

### Renaming a Class

Double-click a class name in the list to rename it.

### Changing a Class Color

Select a class and click **Color** to open the color picker.

## Terrain Profiles

Terrain profiles save your trained classifier and class definitions so you can reuse them on similar imagery without re-labeling.

### Saving a Profile

After a successful classification, use **File > Save Terrain Profile** (Ctrl+S). Profiles are saved as `.cmp` files.

A profile stores:
- Class definitions (names, IDs, colors)
- Trained Random Forest model and feature scaler
- SLIC parameters
- Metadata (source image name, number of labels)

### Loading a Profile

Use **File > Load Terrain Profile** (Ctrl+L). This:
- Replaces the class palette with the profile's class definitions
- Loads the trained model
- Updates SLIC parameters

After loading a profile, open an image, and click **Classify** to apply the saved model without needing to label any training samples. You can also add new labels on top of the loaded model to refine the classification for the specific image.

### Organizing Profiles

Create profiles for each terrain type you work with:
- `nordic_temperate.cmp` — Scandinavian forest/field landscapes
- `arid_desert.cmp` — Desert with sparse vegetation
- `tropical_dense.cmp` — Dense tropical vegetation

## Navigation Controls

| Action | Control |
|--------|---------|
| Pan | Middle-mouse drag, or Ctrl+left-click drag |
| Zoom | Scroll wheel |
| Fit to window | Ctrl+0 |
| Label superpixel | Left-click |
| Remove label | Right-click |

## Tips for Specific Terrain Types

### Nordic/Temperate (Forest + Field)

- Coniferous and deciduous forests often have similar colors but different textures. Label a few examples of each and the texture features (GLCM) will separate them.
- Dead grass and green grass should both be labeled as "Grass/field." The superpixel grouping handles the color variation.
- Toggle superpixel boundaries to check that the treeline-to-field transition is captured by superpixel edges. If not, increase the segment count.

### Desert/Arid

- Bare soil can have significant color variation. Label light and dark soil examples.
- Sparse bushes may be too small for individual superpixels at low segment counts. Increase segments if needed.
- Shadows are minimal in desert scenes, so shadow resolution has less impact.

### Tropical

- Dense canopy may appear uniform. Texture features help distinguish different forest types.
- Water bodies surrounded by dense vegetation may need extra labels to avoid confusion.
- Cloud shadows (if present) are handled by the automatic shadow resolution.

## Using Output in Houdini

### Classified Raster as Heightfield Masks

1. Import the classified GeoTIFF as a Heightfield layer in Houdini.
2. Each unique pixel value corresponds to a class ID. Use the class legend JSON to identify which ID maps to which landcover type.
3. Create masks from each class ID value. For example, `class == 1` gives you the coniferous forest mask.
4. Use these masks to drive material assignment, vegetation scattering, and other procedural operations.

### Density Raster for Vegetation Scatter

1. Import the density GeoTIFF as a separate Heightfield layer.
2. Values range from 0.0 (no canopy) to 1.0 (full canopy cover).
3. Use this directly as a density attribute for scatter operations:
   - Dense forest (0.8-1.0): closely packed trees
   - Sparse forest (0.3-0.5): scattered trees with gaps
   - Open pasture with bushes (0.05-0.2): occasional scattered vegetation
4. Multiply the density by a base scatter density to control absolute tree count.

## Troubleshooting

### Classification is poor

- Add more training labels, especially in misclassified areas.
- Label examples that cover the full range of appearances for each class.
- Check that superpixels are small enough to capture the features you care about.

### Superpixels cross class boundaries

- Increase the segment count (more, smaller superpixels).
- Decrease compactness (segments follow color boundaries more closely).

### Shadow areas are misclassified

- The automatic shadow resolution handles most cases. If shadows are still misclassified, label a few more examples of the correct underlying class near shadowed areas.
- If NIR is available, shadow resolution is significantly better because NDVI distinguishes shadowed vegetation (positive NDVI) from water (near-zero NDVI).

### Application is slow during feature extraction

- Feature extraction computes GLCM texture for every superpixel. With 2000+ segments on a large image, this can take 10-30 seconds. The feature matrix is cached, so subsequent classifications are fast.
- Reducing the segment count speeds up feature extraction linearly.

### Export fails with rasterio error

- Ensure rasterio is installed. PNG-source images can still be exported if rasterio is available; the output GeoTIFF simply won't have CRS information.
