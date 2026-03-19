"""Main application window: orchestrates all GUI components and core pipeline."""

import os

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QSplitter, QVBoxLayout, QWidget,
    QStatusBar, QAction, QMessageBox, QApplication, QPushButton,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from class_maps.config import DEFAULT_CLASSES
from class_maps.core.io_manager import (
    load_image, save_classified_raster, save_density_raster, save_legend_json,
)
from class_maps.core.preprocessor import preprocess_image
from class_maps.core.superpixels import compute_slic, compute_slic_with_linear
from class_maps.core.features import extract_all_features
from class_maps.core.classifier import LandcoverClassifier
from class_maps.core.postprocessor import (
    morphological_cleanup, detect_shadow_segments, resolve_shadows,
    refine_boundaries,
)
from class_maps.core.density import compute_canopy_density
from class_maps.core.terrain_profile import TerrainProfile, save_profile, load_profile
from class_maps.gui.image_canvas import ImageCanvas
from class_maps.gui.controls_panel import ControlsPanel
from class_maps.gui.class_palette import ClassPalette
from class_maps.gui.dialogs import ExportDialog
from class_maps.utils.color_utils import get_class_colors


class SLICWorker(QThread):
    """Worker thread for SLIC computation (with optional linear pre-detection)."""
    finished = pyqtSignal(object, object)  # labels array, linear_segment_ids set
    error = pyqtSignal(str)

    def __init__(self, rgb_array, gray, n_segments, compactness,
                 detect_linear=False, linear_params=None, manual_mask=None):
        super().__init__()
        self.rgb_array = rgb_array
        self.gray = gray
        self.n_segments = n_segments
        self.compactness = compactness
        self.detect_linear = detect_linear
        self.linear_params = linear_params or {}
        self.manual_mask = manual_mask

    def run(self):
        try:
            has_manual = self.manual_mask is not None and self.manual_mask.any()
            if self.detect_linear or has_manual:
                labels, linear_ids = compute_slic_with_linear(
                    self.rgb_array,
                    self.gray,
                    n_segments=self.n_segments,
                    compactness=self.compactness,
                    detect_linear=self.detect_linear,
                    linear_params=self.linear_params,
                    manual_mask=self.manual_mask,
                )
            else:
                labels = compute_slic(
                    self.rgb_array,
                    n_segments=self.n_segments,
                    compactness=self.compactness,
                )
                linear_ids = set()
            self.finished.emit(labels, linear_ids)
        except Exception as e:
            self.error.emit(str(e))


class FeatureWorker(QThread):
    """Worker thread for feature extraction."""
    finished = pyqtSignal(object, object)  # feature_matrix, segment_ids
    progress = pyqtSignal(int, int)  # current, total
    error = pyqtSignal(str)

    def __init__(self, labels, preprocessed):
        super().__init__()
        self.labels = labels
        self.preprocessed = preprocessed

    def run(self):
        try:
            pp = self.preprocessed
            feature_matrix, segment_ids = extract_all_features(
                self.labels,
                pp["rgb"], pp["hsv"], pp["lab"], pp["gray"],
                pp["edges_canny"], ndvi=pp["ndvi"],
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )
            self.finished.emit(feature_matrix, segment_ids)
        except Exception as e:
            self.error.emit(str(e))


class ClassMapsWindow(QMainWindow):
    """Main application window for class_maps."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("class_maps - Landcover Classification")
        self.setMinimumSize(1024, 768)

        # State
        self._image_data = None      # ImageData namedtuple
        self._preprocessed = None    # dict from preprocess_image
        self._labels = None          # superpixel labels array
        self._linear_segment_ids = set()  # segment IDs that are linear features
        self._labeled_segments = {}  # {segment_id: class_id}
        self._class_raster = None    # classification result
        self._density_raster = None  # density result
        self._feature_matrix = None  # cached feature matrix
        self._segment_ids = None     # segment IDs for feature matrix rows
        self._classifier = LandcoverClassifier()
        self._worker = None          # background thread

        self._init_ui()
        self._init_menu()
        self._connect_signals()

    def _init_ui(self):
        """Set up the main UI layout."""
        # Central splitter: canvas (left) + controls (right)
        splitter = QSplitter(Qt.Horizontal)

        self._canvas = ImageCanvas()
        splitter.addWidget(self._canvas)

        # Right sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        # Class palette
        self._palette = ClassPalette()
        sidebar_layout.addWidget(self._palette)

        # Classify button
        self._classify_btn = QPushButton("Classify")
        self._classify_btn.setEnabled(False)
        self._classify_btn.setMinimumHeight(40)
        self._classify_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; }"
        )
        self._classify_btn.clicked.connect(self._run_classification)
        sidebar_layout.addWidget(self._classify_btn)

        # Controls panel
        self._controls = ControlsPanel()
        sidebar_layout.addWidget(self._controls)

        splitter.addWidget(sidebar)

        # Set initial splitter sizes (75% canvas, 25% sidebar)
        splitter.setSizes([750, 250])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready. Open an image to begin.")

    def _init_menu(self):
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Image...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        save_profile_action = QAction("&Save Terrain Profile...", self)
        save_profile_action.setShortcut("Ctrl+S")
        save_profile_action.triggered.connect(self._save_terrain_profile)
        file_menu.addAction(save_profile_action)

        load_profile_action = QAction("&Load Terrain Profile...", self)
        load_profile_action.setShortcut("Ctrl+L")
        load_profile_action.triggered.connect(self._load_terrain_profile)
        file_menu.addAction(load_profile_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self._fit_to_window)
        view_menu.addAction(fit_action)

    def _connect_signals(self):
        """Wire up signals between components."""
        # Controls panel
        self._controls.recompute_requested.connect(self._recompute_superpixels)
        self._controls.boundary_toggled.connect(self._canvas.toggle_boundaries)
        self._controls.classification_toggled.connect(self._canvas.toggle_classification)
        self._controls.density_toggled.connect(self._canvas.toggle_density)
        self._controls.classification_opacity_changed.connect(
            self._on_classification_opacity_changed
        )
        self._controls.density_opacity_changed.connect(
            self._on_density_opacity_changed
        )

        # Tool mode and drawing
        self._controls.tool_mode_changed.connect(self._on_tool_mode_changed)
        self._controls.line_width_changed.connect(self._canvas.set_line_width)
        self._controls.clear_lines_requested.connect(self._on_clear_lines)
        self._controls.undo_line_requested.connect(self._on_undo_line)

        # Canvas
        self._canvas.cursor_moved.connect(self._on_cursor_moved)
        self._canvas.pixel_clicked.connect(self._on_pixel_clicked)
        self._canvas.pixel_right_clicked.connect(self._on_pixel_right_clicked)
        self._canvas.polyline_finished.connect(self._on_polyline_finished)

        # Palette
        self._palette.classes_changed.connect(self._on_classes_changed)

    # --- Image loading ---

    def _open_image(self):
        """Open a file dialog and load an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Satellite Image",
            "",
            "Images (*.png *.tif *.tiff);;All Files (*)",
        )
        if not file_path:
            return

        try:
            self._image_data = load_image(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
            return

        rgb = self._image_data.array[:, :, :3]
        self._canvas.set_image(rgb)

        h, w = rgb.shape[:2]
        nir_str = " + NIR" if self._image_data.has_nir else ""
        self._status_bar.showMessage(
            f"Loaded: {os.path.basename(file_path)} ({w}x{h}{nir_str})"
        )

        # Reset state
        self._labels = None
        self._linear_segment_ids = set()
        self._labeled_segments = {}
        self._class_raster = None
        self._density_raster = None
        self._preprocessed = None
        self._feature_matrix = None
        self._segment_ids = None
        self._classifier = LandcoverClassifier()
        self._classify_btn.setEnabled(False)
        self._controls.enable_classification_overlay(False)
        self._controls.enable_density_overlay(False)
        self._palette.update_labeled_counts({})

        # Auto-compute superpixels
        self._recompute_superpixels()

    # --- Superpixel computation ---

    def _recompute_superpixels(self):
        """Compute SLIC superpixels in a background thread."""
        if self._image_data is None:
            return

        if self._labeled_segments:
            reply = QMessageBox.question(
                self, "Recompute Superpixels",
                "This will clear all labels. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._labeled_segments = {}
        self._class_raster = None
        self._feature_matrix = None
        self._segment_ids = None
        self._linear_segment_ids = set()
        self._controls.enable_classification_overlay(False)
        self._controls.enable_density_overlay(False)
        self._classify_btn.setEnabled(False)
        self._palette.update_labeled_counts({})

        params = self._controls.get_slic_params()
        linear_params = self._controls.get_linear_params()
        rgb = self._image_data.array[:, :, :3]

        # Need grayscale for linear detection — preprocess if not done
        if self._preprocessed is None:
            self._status_bar.showMessage("Preprocessing image...")
            QApplication.processEvents()
            self._preprocessed = preprocess_image(
                self._image_data.array,
                has_nir=self._image_data.has_nir,
            )

        detect_linear = linear_params["enabled"]
        linear_det_params = {}
        if detect_linear:
            linear_det_params["confidence"] = linear_params.get("confidence", 0.5)

        # Build manual mask from drawn polylines
        manual_mask = self._get_manual_linear_mask()
        has_manual = manual_mask is not None and manual_mask.any()

        msg = "Computing superpixels"
        parts = []
        if detect_linear:
            parts.append("auto-detection")
        if has_manual:
            parts.append(f"{len(self._canvas.get_drawn_polylines())} drawn lines")
        if parts:
            msg += " (with " + " + ".join(parts) + ")..."
        else:
            msg += "..."
        self._status_bar.showMessage(msg)
        self._controls.recompute_btn.setEnabled(False)

        self._worker = SLICWorker(
            rgb, self._preprocessed["gray"],
            params["n_segments"], params["compactness"],
            detect_linear=detect_linear,
            linear_params=linear_det_params,
            manual_mask=manual_mask,
        )
        self._worker.finished.connect(self._on_slic_finished)
        self._worker.error.connect(self._on_slic_error)
        self._worker.start()

    def _on_slic_finished(self, labels, linear_ids):
        """Handle SLIC computation completion."""
        self._labels = labels
        self._linear_segment_ids = linear_ids
        self._canvas.set_superpixel_labels(labels)

        n_segments = len(np.unique(labels))
        self._controls.recompute_btn.setEnabled(True)

        # Render boundary overlay
        self._canvas.set_boundary_overlay(labels)

        n_linear = len(linear_ids)
        if n_linear > 0:
            self._status_bar.showMessage(
                f"Ready. {n_segments} segments ({n_linear} linear features detected). "
                f"Select a class and click to label."
            )
        else:
            self._status_bar.showMessage(
                f"Ready. {n_segments} superpixels. Select a class and click to label."
            )
        self._worker = None

    def _on_slic_error(self, error_msg):
        """Handle SLIC computation error."""
        QMessageBox.critical(self, "Error", f"SLIC failed:\n{error_msg}")
        self._controls.recompute_btn.setEnabled(True)
        self._status_bar.showMessage("SLIC computation failed.")
        self._worker = None

    # --- Labeling ---

    def _on_pixel_clicked(self, row, col):
        """Handle left-click: label the superpixel under cursor."""
        if self._labels is None:
            return

        class_id = self._palette.get_active_class_id()
        if class_id is None:
            self._status_bar.showMessage("Select a class first.")
            return

        seg_id = int(self._labels[row, col])
        self._labeled_segments[seg_id] = class_id

        # Update palette counts
        self._palette.update_labeled_counts(self._labeled_segments)

        # Update label feedback overlay
        class_colors = get_class_colors(self._palette.get_class_definitions())
        self._canvas.set_label_feedback(
            self._labels, self._labeled_segments, class_colors
        )

        # Enable classify button if at least 2 classes labeled
        unique_classes = set(self._labeled_segments.values())
        self._classify_btn.setEnabled(len(unique_classes) >= 2)

        class_name = self._palette.get_class_definitions()[class_id]["name"]
        self._status_bar.showMessage(
            f"Labeled segment {seg_id} as '{class_name}' "
            f"({len(self._labeled_segments)} total labels)"
        )

    def _on_pixel_right_clicked(self, row, col):
        """Handle right-click: remove label from superpixel."""
        if self._labels is None:
            return

        seg_id = int(self._labels[row, col])
        if seg_id in self._labeled_segments:
            del self._labeled_segments[seg_id]
            self._palette.update_labeled_counts(self._labeled_segments)

            class_colors = get_class_colors(self._palette.get_class_definitions())
            self._canvas.set_label_feedback(
                self._labels, self._labeled_segments, class_colors
            )

            unique_classes = set(self._labeled_segments.values())
            self._classify_btn.setEnabled(len(unique_classes) >= 2)

            self._status_bar.showMessage(
                f"Removed label from segment {seg_id} "
                f"({len(self._labeled_segments)} total labels)"
            )

    def _on_classes_changed(self):
        """Handle class palette changes (add/remove/rename/recolor)."""
        # Remove labels for deleted classes
        current_ids = set(self._palette.get_class_definitions().keys())
        to_remove = [
            seg_id for seg_id, class_id in self._labeled_segments.items()
            if class_id not in current_ids
        ]
        for seg_id in to_remove:
            del self._labeled_segments[seg_id]

        self._palette.update_labeled_counts(self._labeled_segments)

        # Refresh label feedback overlay
        if self._labels is not None and self._labeled_segments:
            class_colors = get_class_colors(self._palette.get_class_definitions())
            self._canvas.set_label_feedback(
                self._labels, self._labeled_segments, class_colors
            )

    # --- Drawing tools ---

    def _on_tool_mode_changed(self, mode):
        """Handle tool mode change from controls panel."""
        self._canvas.set_mode(mode)
        if mode == "draw_line":
            self._status_bar.showMessage(
                "Draw mode: click to place points, double-click or right-click to finish. "
                "Escape to cancel."
            )
        else:
            self._status_bar.showMessage("Label mode: click superpixels to assign classes.")

    def _on_polyline_finished(self, polyline):
        """Handle a completed polyline from the canvas."""
        n_lines = len(self._canvas.get_drawn_polylines())
        self._controls.update_lines_count(n_lines)
        self._status_bar.showMessage(
            f"Line drawn ({len(polyline)} points). "
            f"{n_lines} total lines. Recompute superpixels to apply."
        )

    def _on_clear_lines(self):
        """Clear all drawn polylines."""
        self._canvas.clear_drawn_lines()
        self._controls.update_lines_count(0)
        self._status_bar.showMessage("All drawn lines cleared.")

    def _on_undo_line(self):
        """Undo the last drawn polyline."""
        self._canvas.undo_last_line()
        n_lines = len(self._canvas.get_drawn_polylines())
        self._controls.update_lines_count(n_lines)
        self._status_bar.showMessage(f"Last line removed. {n_lines} lines remaining.")

    def _get_manual_linear_mask(self):
        """Build a boolean mask from user-drawn polylines.

        Returns None if no polylines have been drawn.
        """
        polylines = self._canvas.get_drawn_polylines()
        if not polylines:
            return None

        from class_maps.core.linear_features import rasterize_polylines
        h, w = self._image_data.array.shape[:2]
        line_width = self._controls.line_width_spin.value()
        return rasterize_polylines(polylines, (h, w), line_width)

    # --- Classification ---

    def _run_classification(self):
        """Run the classification pipeline."""
        if self._labels is None:
            return

        has_labels = bool(self._labeled_segments)
        has_trained_model = self._classifier.is_trained

        if not has_labels and not has_trained_model:
            QMessageBox.warning(
                self, "Classification",
                "Label at least 2 different classes, or load a terrain profile."
            )
            return

        if has_labels:
            unique_classes = set(self._labeled_segments.values())
            if len(unique_classes) < 2 and not has_trained_model:
                QMessageBox.warning(
                    self, "Classification",
                    "Label at least 2 different classes before classifying."
                )
                return

        # Extract features if not cached
        if self._feature_matrix is None:
            self._status_bar.showMessage("Extracting features...")
            self._classify_btn.setEnabled(False)
            QApplication.processEvents()

            self._worker = FeatureWorker(self._labels, self._preprocessed)
            self._worker.finished.connect(self._on_features_finished)
            self._worker.progress.connect(self._on_feature_progress)
            self._worker.error.connect(self._on_feature_error)
            self._worker.start()
        else:
            self._do_classify()

    def _on_feature_progress(self, current, total):
        """Update progress during feature extraction."""
        self._status_bar.showMessage(
            f"Extracting features... {current}/{total} segments"
        )

    def _on_features_finished(self, feature_matrix, segment_ids):
        """Handle feature extraction completion."""
        self._feature_matrix = feature_matrix
        self._segment_ids = list(segment_ids)
        self._worker = None
        self._do_classify()

    def _on_feature_error(self, error_msg):
        """Handle feature extraction error."""
        QMessageBox.critical(self, "Error", f"Feature extraction failed:\n{error_msg}")
        self._classify_btn.setEnabled(True)
        self._worker = None

    def _do_classify(self):
        """Train the classifier, predict, post-process, and compute density."""
        oob_score = None

        if self._labeled_segments:
            # Train on user labels
            self._status_bar.showMessage("Training classifier...")
            QApplication.processEvents()

            try:
                oob_score = self._classifier.train(
                    self._feature_matrix,
                    self._segment_ids,
                    self._labeled_segments,
                )
            except ValueError as e:
                QMessageBox.warning(self, "Classification", str(e))
                self._classify_btn.setEnabled(True)
                return
        elif not self._classifier.is_trained:
            QMessageBox.warning(
                self, "Classification",
                "No labels provided and no profile loaded."
            )
            self._classify_btn.setEnabled(True)
            return
        else:
            # Using a loaded profile without new labels
            oob_score = self._classifier.oob_score

        self._status_bar.showMessage("Predicting...")
        QApplication.processEvents()

        predictions = self._classifier.predict(self._feature_matrix)

        # Build pixel-level class raster
        seg_to_class = {
            self._segment_ids[i]: int(predictions[i])
            for i in range(len(self._segment_ids))
        }
        h, w = self._labels.shape
        self._class_raster = np.zeros((h, w), dtype=np.uint8)
        for seg_id, class_id in seg_to_class.items():
            self._class_raster[self._labels == seg_id] = class_id

        # --- Post-processing ---
        self._status_bar.showMessage("Post-processing: cleanup...")
        QApplication.processEvents()

        self._class_raster = morphological_cleanup(
            self._class_raster, self._labels, min_region_pixels=50
        )

        # Shadow resolution
        self._status_bar.showMessage("Post-processing: resolving shadows...")
        QApplication.processEvents()

        shadow_segs = detect_shadow_segments(
            self._labels, self._preprocessed["hsv"], self._segment_ids
        )
        if shadow_segs:
            seg_id_to_row = {
                sid: i for i, sid in enumerate(self._segment_ids)
            }
            self._class_raster = resolve_shadows(
                self._class_raster, self._labels, self._segment_ids,
                shadow_segs, self._feature_matrix, seg_id_to_row,
            )

        # Boundary refinement
        self._status_bar.showMessage("Post-processing: refining boundaries...")
        QApplication.processEvents()

        self._class_raster = refine_boundaries(
            self._class_raster, self._preprocessed["edges_canny"]
        )

        # --- Density layer ---
        self._status_bar.showMessage("Computing canopy density...")
        QApplication.processEvents()

        # Build segment_classes from the post-processed raster
        segment_classes_final = {}
        for seg_id in self._segment_ids:
            mask = self._labels == seg_id
            classes_in_seg = self._class_raster[mask]
            if len(classes_in_seg) > 0:
                values, counts = np.unique(classes_in_seg, return_counts=True)
                segment_classes_final[seg_id] = int(values[np.argmax(counts)])

        self._density_raster = compute_canopy_density(
            self._labels,
            segment_classes_final,
            self._preprocessed["gray"],
            ndvi=self._preprocessed["ndvi"],
        )

        # --- Render overlays ---
        class_colors = get_class_colors(self._palette.get_class_definitions())
        alpha = self._controls.classification_opacity.value()
        self._canvas.set_classification_overlay(
            self._class_raster, class_colors, alpha
        )

        density_alpha = self._controls.density_opacity.value()
        self._canvas.set_density_overlay(self._density_raster, density_alpha)

        # Enable overlays
        self._controls.enable_classification_overlay(True)
        self._controls.enable_density_overlay(True)
        self._canvas.toggle_classification(True)
        self._controls.classification_check.setChecked(True)
        self._classify_btn.setEnabled(True)

        n_shadow = len(shadow_segs) if shadow_segs else 0
        oob_str = f"OOB: {oob_score:.1%} | " if oob_score is not None else ""
        self._status_bar.showMessage(
            f"Classification complete. {oob_str}"
            f"{len(self._labeled_segments)} labels | "
            f"{n_shadow} shadow segments resolved"
        )

    def _on_classification_opacity_changed(self, alpha):
        """Re-render classification overlay with new opacity."""
        if self._class_raster is not None:
            class_colors = get_class_colors(self._palette.get_class_definitions())
            self._canvas.set_classification_overlay(
                self._class_raster, class_colors, alpha
            )

    def _on_density_opacity_changed(self, alpha):
        """Re-render density overlay with new opacity."""
        if self._density_raster is not None:
            self._canvas.set_density_overlay(self._density_raster, alpha)

    # --- Export ---

    def _export_results(self):
        """Export classified raster, density raster, and legend."""
        if self._class_raster is None:
            QMessageBox.warning(
                self, "Export", "No classification results to export. "
                "Run classification first."
            )
            return

        has_density = self._density_raster is not None
        has_crs = (
            self._image_data is not None and self._image_data.crs is not None
        )

        dialog = ExportDialog(
            self, has_density=has_density, has_crs=has_crs
        )
        if dialog.exec_() != ExportDialog.Accepted:
            return

        settings = dialog.get_export_settings()
        output_dir = settings["output_dir"]
        prefix = settings["prefix"]

        crs = self._image_data.crs if self._image_data else None
        transform = self._image_data.transform if self._image_data else None

        exported = []

        try:
            if settings["export_classified"]:
                path = os.path.join(output_dir, f"{prefix}_classes.tif")
                save_classified_raster(path, self._class_raster, crs, transform)
                exported.append(f"Classes: {path}")

            if settings["export_density"] and self._density_raster is not None:
                path = os.path.join(output_dir, f"{prefix}_density.tif")
                save_density_raster(path, self._density_raster, crs, transform)
                exported.append(f"Density: {path}")

            if settings["export_legend"]:
                path = os.path.join(output_dir, f"{prefix}_legend.json")
                save_legend_json(path, self._palette.get_class_definitions())
                exported.append(f"Legend: {path}")

            self._status_bar.showMessage(
                f"Exported {len(exported)} file(s) to {output_dir}"
            )
            QMessageBox.information(
                self, "Export Complete",
                "Exported:\n" + "\n".join(exported),
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Export failed:\n{e}")

    # --- Status bar ---

    def _on_cursor_moved(self, row, col):
        """Update status bar with cursor position info."""
        parts = [f"Pixel: ({col}, {row})"]

        if self._labels is not None:
            seg_id = int(self._labels[row, col])
            seg_type = " (linear)" if seg_id in self._linear_segment_ids else ""
            parts.append(f"Seg: {seg_id}{seg_type}")
            if seg_id in self._labeled_segments:
                class_id = self._labeled_segments[seg_id]
                class_defs = self._palette.get_class_definitions()
                if class_id in class_defs:
                    parts.append(f"Label: {class_defs[class_id]['name']}")

        if self._image_data is not None:
            rgb = self._image_data.array[row, col, :3]
            parts.append(f"RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]})")

        if self._class_raster is not None:
            cid = self._class_raster[row, col]
            class_defs = self._palette.get_class_definitions()
            if cid in class_defs:
                parts.append(f"Class: {class_defs[cid]['name']}")

        self._status_bar.showMessage("  |  ".join(parts))

    def _fit_to_window(self):
        """Fit the image to the window."""
        if self._canvas._image_item is not None:
            self._canvas.fitInView(
                self._canvas._image_item, Qt.KeepAspectRatio
            )

    # --- Terrain Profiles ---

    def _save_terrain_profile(self):
        """Save current classifier and class definitions as a terrain profile."""
        if not self._classifier.is_trained:
            QMessageBox.warning(
                self, "Save Profile",
                "No trained classifier to save. Run classification first."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Terrain Profile", "",
            "Class Maps Profile (*.cmp);;All Files (*)",
        )
        if not path:
            return

        if not path.endswith(".cmp"):
            path += ".cmp"

        model, scaler = self._classifier.get_model_and_scaler()
        slic_params = self._controls.get_slic_params()

        source_name = ""
        if self._image_data and self._image_data.path:
            source_name = os.path.basename(self._image_data.path)

        profile = TerrainProfile(
            class_definitions=dict(self._palette.get_class_definitions()),
            model=model,
            scaler=scaler,
            slic_params=slic_params,
            metadata={
                "source_image": source_name,
                "n_labeled": str(len(self._labeled_segments)),
            },
            drawn_polylines=[list(p) for p in self._canvas.get_drawn_polylines()],
            line_width=self._controls.line_width_spin.value(),
        )

        try:
            save_profile(path, profile)
            self._status_bar.showMessage(f"Profile saved: {path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Save Error", f"Failed to save profile:\n{e}"
            )

    def _load_terrain_profile(self):
        """Load a terrain profile and optionally classify current image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Terrain Profile", "",
            "Class Maps Profile (*.cmp);;All Files (*)",
        )
        if not path:
            return

        try:
            profile = load_profile(path)
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error", f"Failed to load profile:\n{e}"
            )
            return

        # Update class palette
        self._palette.set_class_definitions(profile.class_definitions)

        # Update SLIC params
        if "n_segments" in profile.slic_params:
            self._controls.n_segments_spin.setValue(
                int(profile.slic_params["n_segments"])
            )
        if "compactness" in profile.slic_params:
            self._controls.compactness_spin.setValue(
                float(profile.slic_params["compactness"])
            )

        # Load the trained model
        self._classifier.set_model_and_scaler(profile.model, profile.scaler)

        # Load drawn polylines
        if profile.drawn_polylines:
            self._canvas.set_drawn_polylines(profile.drawn_polylines)
            self._controls.line_width_spin.setValue(profile.line_width)
            self._canvas.set_line_width(profile.line_width)
            self._controls.update_lines_count(len(profile.drawn_polylines))

        source = profile.metadata.get("source_image", "unknown")
        n_lines = len(profile.drawn_polylines)
        lines_str = f", {n_lines} drawn lines" if n_lines else ""
        self._status_bar.showMessage(
            f"Profile loaded (trained on: {source}{lines_str}). "
            f"Open an image and click Classify to apply."
        )

        # If we have features, enable classify button
        if self._feature_matrix is not None:
            self._classify_btn.setEnabled(True)
