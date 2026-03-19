# class_maps — Project Plan

## Project Summary

Semi-supervised landcover classification tool that takes satellite imagery (PNG/GeoTIFF, 3-4 bands) and produces classified rasters + canopy density layers for Houdini-based 3D scene generation in EO/IR sensor simulation workflows.

---

## Completed Work

### Phase 1: Foundation — Image I/O, Preprocessing, GUI Shell
- [x] Image loading (PNG + GeoTIFF with CRS preservation)
- [x] Color space preprocessing (HSV, Lab, grayscale, NDVI)
- [x] Edge detection (Canny, Sobel)
- [x] SLIC superpixel segmentation
- [x] PyQt5 GUI with pan/zoom/overlay compositing
- [x] Controls panel (SLIC params, overlay toggles, opacity sliders)

### Phase 2: Labeling and Classification
- [x] Click-to-label superpixels (left=assign, right=remove)
- [x] Class palette widget (add/remove/rename/recolor classes)
- [x] 30-dimension feature extraction per superpixel (RGB/HSV/Lab stats, GLCM texture, NDVI, edge density, shape)
- [x] Random Forest classifier with OOB accuracy
- [x] Classification overlay rendering
- [x] Threaded workers for SLIC and feature extraction

### Phase 3: Post-processing, Density, and Export
- [x] Morphological cleanup (small region removal)
- [x] Automatic shadow detection and resolution
- [x] Edge-aware boundary refinement
- [x] Canopy density layer (texture + NDVI)
- [x] Export dialog (GeoTIFF + legend JSON)

### Phase 4: Terrain Profiles
- [x] Save/load `.cmp` terrain profiles (JSON metadata + pickled model/scaler)
- [x] Labeled count per class in palette
- [x] Status bar with segment info on hover

### Documentation
- [x] Installation guide (`docs/install_guide.md`)
- [x] User's guide (`docs/users_guide.md`)
- [x] Technical description (`docs/technical_description.md`)

---

## Current Work

### Linear Feature Pre-Detection (roads, rivers, paths)

**Problem:** SLIC superpixels absorb narrow linear features into surrounding terrain, mixing road pixels with adjacent fields/forest. Roads and rivers need their own dedicated segments.

**Status:** U-Net infrastructure complete. Heuristic fallback in place. **Next step: train the model on the Massachusetts Roads Dataset.**

#### Completed: U-Net Infrastructure
- [x] Research pre-trained models (none available in PyTorch for satellite roads)
- [x] Add PyTorch + segmentation-models-pytorch dependencies
- [x] Create `road_model.py` — U-Net creation, weight loading, tile-based inference with overlap blending
- [x] Create `train_road_model.py` — training script with dataset download, BCE+Dice loss, validation
- [x] Integrate into `linear_features.py` — auto-detects trained weights, falls back to heuristic
- [x] Update GUI — model status indicator, confidence threshold control
- [x] Verify full pipeline (model creation, training loop, inference all tested)

#### Next: Train the Model
To train, run:
```bash
python -m class_maps.train_road_model
```
This downloads the Massachusetts Roads Dataset (~5.8 GB) and trains for 25 epochs.
- GPU: ~15 minutes
- CPU: ~1 hour
- Trained weights saved to `~/.class_maps/models/road_unet_resnet34.pth`
- If the original server is unavailable, download from Kaggle manually (instructions printed by script)

---

## Future Work

### Manual Linear Feature Drawing
- [ ] Add "Draw Linear Features" tool mode to the canvas (click to place points, double-click to finish polyline)
- [ ] Width spinner in controls panel (default ~10px, adjustable per feature)
- [ ] Rasterize drawn polylines into boolean mask, feed into `compute_slic_with_linear()` same as U-Net/heuristic output
- [ ] Combine with U-Net/heuristic masks (OR together) so user can supplement auto-detection
- [ ] Save/load drawn polylines with terrain profile (`.cmp`) for reuse across similar images
- [ ] Support for different linear feature types (road, river, path, fence line, airstrip, etc.)

### Linear Feature Fine-Tuning (Option B)
- [ ] Create a lightweight fine-tuning script for user-specific terrain
- [ ] Allow labeling road pixels directly in the GUI for fine-tuning data
- [ ] Save fine-tuned weights as part of terrain profile (`.cmp`)
- [ ] Support different linear feature types (paved road, gravel road, river, path)

### Phase 5: Large Image Support
- [ ] Tiled processing (2048x2048 tiles with 256px overlap)
- [ ] Windowed GeoTIFF reading via rasterio
- [ ] Lazy tile-based canvas rendering
- [ ] Memory bounded to ~2 tiles at a time
- [ ] Classification remains global (single RF across all tiles)

### Quality of Life
- [ ] Undo stack for labeling operations
- [ ] Keyboard shortcuts for common classes
- [ ] Batch export (multiple images with same profile)
- [ ] Progress bars for long operations
- [ ] Auto-save labeled segments periodically

### Classification Improvements
- [ ] Confidence map overlay (highlight uncertain segments for review)
- [ ] Active learning: suggest which segments to label next for max improvement
- [ ] Multi-class density layers (not just canopy — e.g., building density)
- [ ] Support for additional spectral bands beyond NIR

---

## Architecture

```
Load Image → Pre-process → [U-Net Linear Detection] → SLIC Superpixels →
Feature Extraction → User Labels ROIs → Random Forest Classification →
Post-processing → Density Layer → Export (GeoTIFF + Legend)
```

### Tech Stack
- **Core:** numpy, scikit-image, scikit-learn, opencv-python, rasterio
- **GUI:** PyQt5
- **Linear detection:** PyTorch, segmentation-models-pytorch (new)
- **Image I/O:** Pillow, rasterio

### Key Files
| File | Purpose |
|------|---------|
| `run.py` | Entry point |
| `class_maps/config.py` | Default classes, SLIC/RF params, constants |
| `class_maps/core/linear_features.py` | Linear feature pre-detection (roads/rivers) |
| `class_maps/core/road_model.py` | U-Net model creation, weight loading, tile inference |
| `class_maps/train_road_model.py` | Training script for road segmentation model |
| `class_maps/core/superpixels.py` | SLIC + linear feature merging |
| `class_maps/core/features.py` | 30-dim feature extraction per superpixel |
| `class_maps/core/classifier.py` | Random Forest train/predict |
| `class_maps/core/postprocessor.py` | Morphological cleanup, shadow resolution |
| `class_maps/core/density.py` | Canopy density layer |
| `class_maps/core/terrain_profile.py` | Save/load trained models |
| `class_maps/core/io_manager.py` | GeoTIFF/PNG I/O with CRS |
| `class_maps/gui/main_window.py` | Main orchestrator |
| `class_maps/gui/image_canvas.py` | Pan/zoom/click-to-label canvas |
| `class_maps/gui/class_palette.py` | Class list sidebar |
| `class_maps/gui/controls_panel.py` | Parameter controls |
| `class_maps/gui/overlay_renderer.py` | Overlay compositing |
| `class_maps/gui/dialogs.py` | Export dialog |
