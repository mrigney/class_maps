"""Controls panel: SLIC parameters, overlay toggles, opacity sliders."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QSlider, QCheckBox, QGroupBox,
)
from PyQt5.QtCore import Qt, pyqtSignal

from class_maps.config import SLIC_N_SEGMENTS, SLIC_COMPACTNESS


class ControlsPanel(QWidget):
    """Sidebar panel for SLIC parameters and overlay controls."""

    recompute_requested = pyqtSignal()
    boundary_toggled = pyqtSignal(bool)
    classification_toggled = pyqtSignal(bool)
    density_toggled = pyqtSignal(bool)
    classification_opacity_changed = pyqtSignal(int)
    density_opacity_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # --- SLIC Parameters ---
        slic_group = QGroupBox("Superpixel Parameters")
        slic_layout = QVBoxLayout()

        # N segments
        n_seg_layout = QHBoxLayout()
        n_seg_layout.addWidget(QLabel("Segments:"))
        self.n_segments_spin = QSpinBox()
        self.n_segments_spin.setRange(100, 20000)
        self.n_segments_spin.setValue(SLIC_N_SEGMENTS)
        self.n_segments_spin.setSingleStep(100)
        n_seg_layout.addWidget(self.n_segments_spin)
        slic_layout.addLayout(n_seg_layout)

        # Compactness
        compact_layout = QHBoxLayout()
        compact_layout.addWidget(QLabel("Compactness:"))
        self.compactness_spin = QDoubleSpinBox()
        self.compactness_spin.setRange(1.0, 100.0)
        self.compactness_spin.setValue(SLIC_COMPACTNESS)
        self.compactness_spin.setSingleStep(1.0)
        compact_layout.addWidget(self.compactness_spin)
        slic_layout.addLayout(compact_layout)

        # Linear feature pre-detection
        self.linear_check = QCheckBox("Detect linear features (roads/rivers)")
        self.linear_check.setChecked(True)
        self.linear_check.setToolTip(
            "Detect roads, rivers, and paths before superpixel segmentation.\n"
            "Gives linear features their own dedicated segments instead of\n"
            "letting them bleed into surrounding terrain."
        )
        slic_layout.addWidget(self.linear_check)

        # U-Net model status
        self._unet_status_label = QLabel("  Model: checking...")
        self._unet_status_label.setStyleSheet("color: gray; font-size: 11px;")
        slic_layout.addWidget(self._unet_status_label)
        self._check_unet_status()

        # Confidence threshold (used by U-Net, replaces old ridge sensitivity)
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("  Confidence:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.10, 0.95)
        self.confidence_spin.setValue(0.50)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setToolTip(
            "Road detection confidence threshold.\n"
            "Lower = more sensitive (detects fainter roads).\n"
            "Higher = more selective (only confident detections)."
        )
        confidence_layout.addWidget(self.confidence_spin)
        slic_layout.addLayout(confidence_layout)

        # Keep ridge_threshold_spin as hidden for API compat
        self.ridge_threshold_spin = QDoubleSpinBox()
        self.ridge_threshold_spin.setRange(0.05, 0.50)
        self.ridge_threshold_spin.setValue(0.15)
        self.ridge_threshold_spin.hide()

        # Recompute button
        self.recompute_btn = QPushButton("Recompute Superpixels")
        self.recompute_btn.clicked.connect(self.recompute_requested.emit)
        slic_layout.addWidget(self.recompute_btn)

        slic_group.setLayout(slic_layout)
        layout.addWidget(slic_group)

        # --- Overlay Toggles ---
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QVBoxLayout()

        self.boundary_check = QCheckBox("Superpixel Boundaries")
        self.boundary_check.setChecked(False)
        self.boundary_check.toggled.connect(self.boundary_toggled.emit)
        overlay_layout.addWidget(self.boundary_check)

        self.classification_check = QCheckBox("Classification")
        self.classification_check.setChecked(True)
        self.classification_check.setEnabled(False)  # Enabled after classification
        self.classification_check.toggled.connect(self.classification_toggled.emit)
        overlay_layout.addWidget(self.classification_check)

        # Classification opacity
        class_opacity_layout = QHBoxLayout()
        class_opacity_layout.addWidget(QLabel("  Opacity:"))
        self.classification_opacity = QSlider(Qt.Horizontal)
        self.classification_opacity.setRange(0, 255)
        self.classification_opacity.setValue(128)
        self.classification_opacity.valueChanged.connect(
            self.classification_opacity_changed.emit
        )
        class_opacity_layout.addWidget(self.classification_opacity)
        overlay_layout.addLayout(class_opacity_layout)

        self.density_check = QCheckBox("Canopy Density")
        self.density_check.setChecked(False)
        self.density_check.setEnabled(False)  # Enabled after density computed
        self.density_check.toggled.connect(self.density_toggled.emit)
        overlay_layout.addWidget(self.density_check)

        # Density opacity
        density_opacity_layout = QHBoxLayout()
        density_opacity_layout.addWidget(QLabel("  Opacity:"))
        self.density_opacity = QSlider(Qt.Horizontal)
        self.density_opacity.setRange(0, 255)
        self.density_opacity.setValue(128)
        self.density_opacity.valueChanged.connect(
            self.density_opacity_changed.emit
        )
        density_opacity_layout.addWidget(self.density_opacity)
        overlay_layout.addLayout(density_opacity_layout)

        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)

        layout.addStretch()

    def get_slic_params(self):
        """Return current SLIC parameters as a dict."""
        return {
            "n_segments": self.n_segments_spin.value(),
            "compactness": self.compactness_spin.value(),
        }

    def get_linear_params(self):
        """Return linear feature detection parameters."""
        return {
            "enabled": self.linear_check.isChecked(),
            "confidence": self.confidence_spin.value(),
            "ridge_threshold": self.ridge_threshold_spin.value(),
        }

    def _check_unet_status(self):
        """Check if U-Net model weights are available and update label."""
        try:
            from class_maps.core.linear_features import has_unet_model
            if has_unet_model():
                self._unet_status_label.setText("  Model: U-Net (trained)")
                self._unet_status_label.setStyleSheet("color: green; font-size: 11px;")
            else:
                self._unet_status_label.setText("  Model: heuristic (no trained U-Net)")
                self._unet_status_label.setStyleSheet("color: orange; font-size: 11px;")
        except Exception:
            self._unet_status_label.setText("  Model: heuristic (fallback)")
            self._unet_status_label.setStyleSheet("color: orange; font-size: 11px;")

    def enable_classification_overlay(self, enabled=True):
        """Enable/disable the classification overlay checkbox."""
        self.classification_check.setEnabled(enabled)

    def enable_density_overlay(self, enabled=True):
        """Enable/disable the density overlay checkbox."""
        self.density_check.setEnabled(enabled)
