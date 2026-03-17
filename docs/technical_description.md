# class_maps Technical Description

## Purpose

class_maps is a semi-supervised landcover classification tool designed to produce labeled terrain rasters from satellite imagery. The output drives procedural scene generation in Houdini for EO/IR sensor simulation. The tool prioritizes *general terrain character* over pixel-perfect accuracy, producing spatially coherent classification maps suitable for material assignment and vegetation scattering.

## Architecture Overview

```
Input Image (PNG / GeoTIFF)
        |
        v
  +-----------+     +----------------+     +-------------------+
  | Load &    | --> | SLIC Superpixel| --> | Feature Extraction|
  | Preprocess|     | Segmentation   |     | (per superpixel)  |
  +-----------+     +----------------+     +-------------------+
                                                    |
                                                    v
                                           +------------------+
                          User Labels ---> | Random Forest    |
                          (GUI clicks)     | Classification   |
                                           +------------------+
                                                    |
                                                    v
                                           +------------------+
                                           | Post-processing  |
                                           | (cleanup, shadow |
                                           |  resolution,     |
                                           |  edge refinement)|
                                           +------------------+
                                                    |
                                            +-------+-------+
                                            |               |
                                            v               v
                                     Class Raster    Density Raster
                                     (uint8 TIFF)    (float32 TIFF)
```

## Module Reference

### class_maps/config.py

Central configuration file. All tunable parameters are defined here rather than scattered across modules.

**Key parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `SLIC_N_SEGMENTS` | 2000 | Target number of superpixels |
| `SLIC_COMPACTNESS` | 10.0 | Color vs. spatial weight in SLIC |
| `SLIC_SIGMA` | 1.0 | Gaussian pre-smoothing for SLIC |
| `GLCM_DISTANCES` | [1, 3] | Pixel distances for GLCM computation |
| `GLCM_ANGLES` | [0, 45, 90, 135 deg] | Angles for GLCM computation |
| `RF_N_ESTIMATORS` | 200 | Number of trees in Random Forest |
| `RF_MIN_SAMPLES_LEAF` | 2 | Minimum samples per leaf node |
| `SHADOW_V_THRESHOLD` | 60 | HSV Value threshold for shadow detection |
| `SHADOW_S_THRESHOLD` | 40 | HSV Saturation threshold for shadow detection |
| `VEGETATION_CLASS_IDS` | [1, 2, 3] | Classes eligible for density computation |

### class_maps/core/io_manager.py

Handles all image I/O. Returns a uniform `ImageData` named tuple regardless of input format.

**ImageData fields:**
- `array`: numpy uint8 array, shape (H, W, C) where C is 3 (RGB) or 4 (RGB+NIR)
- `crs`: rasterio CRS object or None (PNG inputs)
- `transform`: rasterio Affine transform or None
- `path`: source file path
- `has_nir`: boolean indicating presence of a 4th NIR band

**Normalization:** GeoTIFF inputs may be uint16 or float32. The loader normalizes to uint8 using per-band 2nd-98th percentile stretching to handle varying dynamic ranges without clipping outliers.

**Export functions:** `save_classified_raster()` and `save_density_raster()` write single-band GeoTIFFs with optional CRS and transform. `save_legend_json()` writes a class ID to name/color mapping.

### class_maps/core/preprocessor.py

Transforms the input RGB image into multiple representations used by feature extraction.

**Outputs:**
- `rgb`: original RGB (H, W, 3) uint8
- `hsv`: Hue-Saturation-Value (H, W, 3) uint8 via OpenCV
- `lab`: CIE L\*a\*b\* (H, W, 3) uint8 via OpenCV
- `gray`: grayscale (H, W) uint8
- `ndvi`: Normalized Difference Vegetation Index (H, W) float32, or None if no NIR band. Computed as `(NIR - Red) / (NIR + Red)`.
- `edges_canny`: binary edge map (H, W) uint8 from Canny detector (thresholds 50/150)
- `edges_sobel`: gradient magnitude (H, W) float32 from Sobel operator

**Rationale for multiple color spaces:**
- RGB captures raw spectral information but is sensitive to illumination.
- HSV separates chromatic content (H, S) from brightness (V), making it useful for shadow detection (shadows have low V but may retain H/S of underlying material).
- CIE Lab is perceptually uniform, meaning Euclidean distances in Lab space correlate better with human-perceived color differences. This helps the classifier learn class boundaries that correspond to visually distinct materials.

### class_maps/core/superpixels.py

Implements SLIC (Simple Linear Iterative Clustering) superpixel segmentation via scikit-image.

**Why superpixels?**

Superpixels are the fundamental design decision in class_maps. Instead of classifying individual pixels (25 million for a 5000x5000 image), we classify superpixels (~2000-5000 segments). This provides:

1. **Spatial coherence**: nearby pixels with similar colors are grouped together, preventing the "salt-and-pepper" noise typical of per-pixel classification.
2. **Computational efficiency**: feature extraction, training, and prediction operate on thousands of segments rather than millions of pixels. Interactive reclassification takes milliseconds.
3. **Meaningful texture computation**: GLCM texture features require a spatial neighborhood. Superpixels provide a natural, boundary-respecting neighborhood.
4. **User interaction**: clicking a single superpixel labels hundreds of pixels at once, making interactive labeling practical.

**SLIC parameters:**

- `n_segments` controls granularity. At 0.3-1.0 m/pixel resolution, 2000 segments on a 5000x5000 image produces segments of roughly 50x50 pixels (15-50 meters on a side), appropriate for distinguishing individual tree stands, field boundaries, and road widths.
- `compactness` trades off color similarity vs. spatial regularity. The default of 10.0 produces reasonably regular shapes while still respecting strong color boundaries like treelines.

### class_maps/core/features.py

Extracts a 30-dimensional feature vector for each superpixel. This is the core of the classification system and the primary mechanism for distinguishing materials with similar RGB values.

**Feature vector composition:**

| Index | Features | Dims | Purpose |
|-------|----------|------|---------|
| 0-5 | RGB mean, std | 6 | Raw spectral information |
| 6-11 | HSV mean, std | 6 | Illumination-invariant color + brightness |
| 12-17 | Lab mean, std | 6 | Perceptually uniform color differences |
| 18-19 | NDVI mean, std | 2 | Vegetation vigor (0 if no NIR) |
| 20-25 | GLCM texture | 6 | Spatial texture pattern |
| 26 | Edge density | 1 | Boundary/transition zone indicator |
| 27-29 | Shape metrics | 3 | Segment geometry |

**GLCM (Gray-Level Co-occurrence Matrix) texture features:**

GLCM captures the spatial arrangement of pixel intensities within each superpixel. For each segment, the grayscale patch within its bounding box is extracted, quantized to 64 levels, and the GLCM is computed at distances [1, 3] pixels and angles [0, 45, 90, 135] degrees. Six properties are extracted and averaged over all distance-angle combinations:

- **Contrast**: measures intensity differences between neighboring pixels. High for rough textures (tree canopy), low for smooth surfaces (water, mowed grass).
- **Dissimilarity**: similar to contrast but with linear weighting. Provides complementary texture information.
- **Homogeneity**: inverse of contrast. High for uniform regions, low for textured regions.
- **Energy** (angular second moment): measures textural uniformity. High for solid-color regions, low for complex textures.
- **Correlation**: measures linear predictability of intensity patterns. Captures directional texture.
- **Entropy**: computed from the normalized GLCM distribution. Measures texture randomness. Dense forest canopy has high entropy; uniform surfaces have low entropy.

**Why GLCM solves the grass-vs-canopy problem:**

Grass and tree canopy often have overlapping mean RGB values, especially in temperate climates where both are green. However, their *textures* are fundamentally different:
- Grass fields are smooth at 0.3-1.0 m resolution: low contrast, high homogeneity, low entropy.
- Tree canopy is rough: individual crowns create intensity variation, producing high contrast, low homogeneity, and high entropy.

The classifier learns these texture signatures from the labeled examples and applies them to all unlabeled superpixels.

**Shape features:**

- **Area (normalized)**: segment pixel count divided by total image pixels. Helps distinguish natural (irregular) from man-made (regular) regions.
- **Compactness**: isoperimetric quotient (4*pi*area / perimeter^2). Circles = 1.0, elongated shapes approach 0. Roads produce elongated superpixels with low compactness.
- **Eccentricity**: from second-order central moments of the segment mask. Captures elongation direction.

### class_maps/core/classifier.py

Wraps scikit-learn's `RandomForestClassifier` with a labeling interface for interactive use.

**Why Random Forest?**

1. **Works well with small training sets**: RF can learn from as few as 10-20 labeled superpixels per class, which is realistic for an interactive workflow.
2. **Handles mixed feature types**: the feature vector includes color intensities, ratios, texture statistics, and shape metrics at different scales. RF handles this without requiring feature normalization (though we apply `StandardScaler` anyway for consistency).
3. **Out-of-bag (OOB) error estimate**: each tree is trained on a bootstrap sample, and the remaining samples provide a built-in cross-validation estimate. This gives the user an accuracy metric without needing a separate validation set.
4. **Feature importance**: RF provides a ranking of which features are most discriminative, useful for debugging classification issues.
5. **Fast training and prediction**: with ~2000 segments and 30 features, training 200 trees takes <1 second. Prediction is effectively instantaneous.

**Training protocol:**

1. `StandardScaler` is fit on *all* superpixel features (not just labeled ones) to capture the full feature distribution.
2. Labeled segments are extracted and used to train the RF.
3. OOB score is computed and returned as a quality metric.
4. All segments are predicted in a single batch.

**Parameters:**
- 200 estimators (trees) provides a good accuracy/speed tradeoff.
- `min_samples_leaf=2` prevents overfitting on small training sets.
- `oob_score=True` enables the OOB accuracy estimate.
- `n_jobs=-1` uses all CPU cores for parallel tree construction.

### class_maps/core/postprocessor.py

Cleans up the raw classification output through three stages:

**1. Morphological cleanup** (`morphological_cleanup`):

Removes small isolated classified regions (fewer than 50 pixels by default) by majority vote from neighboring regions. Uses scipy `ndimage.label` to identify connected components per class, then dilates each small region to find its neighbors and assigns it to the most common neighboring class.

This eliminates the scattered misclassifications that can occur when a single superpixel has ambiguous features.

**2. Shadow detection and resolution** (`detect_shadow_segments`, `resolve_shadows`):

Shadows are not a meaningful landcover class for scene generation purposes. Instead of requiring users to label shadows (tedious and uninformative), class_maps detects them automatically and reassigns them.

Detection: a superpixel is flagged as shadow if its mean HSV Value < 60 and mean Saturation < 40. These thresholds (configurable in `config.py`) identify dark, desaturated pixels that are characteristic of cast shadows.

Resolution: each shadow superpixel is compared to its non-shadow neighbors using Euclidean distance in feature space. It is reassigned to the class of the most similar neighbor. The intuition is that a shadow on grass still has grass-like texture, and a shadow in forest still has forest-like texture, even though the brightness has changed.

**Why this works better than a "shadow" class:** shadow appearance varies enormously depending on the underlying material, sun angle, and atmospheric conditions. A single "shadow" class would need to encompass shadowed grass, shadowed pavement, shadowed water, etc., which have nothing in common except low brightness. The feature-based neighbor matching correctly recovers the underlying material.

**3. Boundary refinement** (`refine_boundaries`):

Classification boundaries from superpixels are inherently approximate because superpixels are irregularly shaped. This step snaps classification boundaries toward detected image edges (from Canny) where they nearly coincide.

In zones where both a classification boundary and an image edge exist within 2 pixels, a local majority vote in a 5x5 window smooths the transition, producing cleaner boundaries that follow the actual terrain features.

### class_maps/core/density.py

Computes a continuous canopy density raster (0.0-1.0) for vegetation superpixels.

**Algorithm:**

For each superpixel classified as vegetation (class IDs in `VEGETATION_CLASS_IDS`):

1. **Texture score**: standard deviation of grayscale values within the segment, normalized by 128. Dense canopy has more intensity variation (light hitting upper crowns, shadows between crowns) than sparse vegetation.

2. **NDVI score** (if NIR available): mean NDVI within the segment, clipped to [0, 1]. Higher NDVI indicates more photosynthetically active vegetation.

3. **Combined score**: weighted average. With NDVI: 40% texture + 60% NDVI. Without NDVI: 100% texture.

4. **Normalization**: all vegetation segment scores are min-max normalized using the 5th-95th percentile to [0, 1], avoiding outlier influence.

Non-vegetation superpixels receive density = 0.0.

**Downstream use in Houdini:**

The density raster is loaded as a heightfield layer and used directly as a scatter density attribute. A dense coniferous stand might have density 0.85, meaning 85% of the maximum scatter density. A pasture with scattered bushes might have density 0.15. This produces naturalistic vegetation distributions without requiring manual density painting.

### class_maps/core/terrain_profile.py

Serializes and deserializes terrain profiles for reuse across images of similar terrain.

**File format:** `.cmp` files are ZIP archives containing:

| Entry | Format | Content |
|-------|--------|---------|
| `metadata.json` | JSON | Version, creation date, class definitions (names, IDs, colors), SLIC parameters, user metadata |
| `model.pkl` | Python pickle | Trained `RandomForestClassifier` |
| `scaler.pkl` | Python pickle | Fitted `StandardScaler` |

The JSON metadata is human-readable and can be inspected without loading the Python objects. The pickle files require the same major version of scikit-learn that created them.

**Use case:** when working with a cluster of similar terrain images (e.g., multiple scenes from the same Nordic region), train on one image, save the profile, and apply it to the others. The classifier generalizes because the feature space (color, texture, shape) is similar across images of the same terrain type. Users can add corrective labels on top of a loaded profile to handle image-specific variations.

### class_maps/gui/

The GUI is built with PyQt5 and follows a standard QMainWindow layout:

```
+-----------------------------------------------+
| Menu Bar (File, View)                          |
+---------------------------+-------------------+
|                           | Class Palette     |
|                           |   (class list,    |
|                           |    add/remove)    |
|   Image Canvas            |                   |
|   (QGraphicsView)         | [Classify Button] |
|                           |                   |
|   - pan/zoom              | Controls Panel    |
|   - click-to-label        |   (SLIC params,   |
|   - overlay compositing   |    overlay toggles,|
|                           |    opacity sliders)|
+---------------------------+-------------------+
| Status Bar (pixel coords, segment ID, RGB, class) |
+-----------------------------------------------+
```

**Threading model:**

SLIC computation and feature extraction run in `QThread` workers to keep the GUI responsive. The main thread handles user interaction, classification (fast enough to run synchronously), and overlay rendering.

**Overlay compositing:**

The canvas uses `QGraphicsScene` with multiple `QGraphicsPixmapItem` layers at different Z-values:

| Z-order | Layer |
|---------|-------|
| 0 | Base satellite image |
| 10 | Superpixel boundary overlay |
| 15 | Label feedback overlay (user-labeled segments) |
| 20 | Classification overlay |
| 25 | Density overlay |

All overlays are RGBA images with per-pixel alpha, allowing smooth transparency blending.

### class_maps/utils/

**color_utils.py**: Class color management. `generate_distinct_color()` uses golden-ratio hue stepping to produce maximally separated colors when users add new classes.

**geometry_utils.py**: Segment shape metrics. `compute_perimeter()` estimates perimeter by counting boundary pixels (pixels with at least one 4-connected neighbor outside the mask). `compute_eccentricity()` uses second-order central moments to determine segment elongation.

## Shadow vs. Water Discrimination

This is a key challenge addressed by the multi-feature approach. At 0.3-1.0 m/pixel, shadows and water can have overlapping RGB values (both dark). class_maps uses four discriminators:

| Feature | Shadow | Water |
|---------|--------|-------|
| GLCM texture | Retains underlying texture (forest canopy, grass blades) | Very smooth (low contrast, high homogeneity) |
| NDVI (if available) | Positive (vegetation underneath) | Near-zero or negative |
| HSV Hue | Variable (reflects underlying material) | Tends toward blue hue |
| Shape (compactness) | Elongated, adjacent to tall objects | Compact, smooth-edged |

The Random Forest learns to weight these features from the user's labeled examples. In practice, labeling 3-5 water superpixels and 3-5 shadowed forest/grass superpixels is sufficient for the classifier to separate them reliably.

## Spatial Coherence Design

A primary design goal is that small color variations within a single material class should not fragment the classification. For example, a field containing both green and dormant brown grass should be classified uniformly as "Grass/field."

This is achieved at three levels:

1. **Superpixels**: SLIC groups spatially adjacent pixels with similar colors into segments. A field with gradual color variation becomes a single segment (or a few large segments), not thousands of individual pixels.

2. **Feature aggregation**: features are computed as mean and standard deviation *within* each superpixel. The std captures within-segment variation, but the mean still represents the dominant color. Two adjacent superpixels with slightly different grass tones will have similar mean color and similar texture, and will be classified the same way.

3. **Morphological cleanup**: after classification, small isolated regions of a different class are merged into the surrounding class by majority vote.

## Performance Characteristics

| Operation | Time (5000x5000 image, 2000 segments) |
|-----------|---------------------------------------|
| Image load (PNG) | <1 second |
| Preprocessing (HSV, Lab, edges) | ~2 seconds |
| SLIC segmentation | ~5-10 seconds |
| Feature extraction | ~10-30 seconds |
| RF training (200 trees) | <1 second |
| Prediction | <0.1 seconds |
| Post-processing | ~2-5 seconds |
| Density computation | ~1-2 seconds |
| Total (first classification) | ~20-50 seconds |
| Reclassification (cached features) | ~3-6 seconds |

Feature extraction dominates the first classification because GLCM computation is per-segment. Once cached, reclassification is fast, enabling rapid iterative refinement.

## Limitations and Future Work

**Current limitations:**
- Maximum practical image size is ~10000x10000 pixels without tiled processing. Memory usage scales linearly with pixel count.
- Terrain profiles may not transfer well between vastly different imagery (different sensors, resolutions, or atmospheric conditions) because the feature space shifts.
- GLCM texture is resolution-dependent: features learned at 0.3 m/pixel may not generalize to 1.0 m/pixel imagery of the same terrain.

**Planned (Phase 5):**
- Tiled processing for images up to 40000x40000 pixels using overlapping tiles with stitched classification.
- Windowed GeoTIFF reading via rasterio for memory-bounded loading.
- Lazy tile-based canvas rendering for smooth pan/zoom on large images.
