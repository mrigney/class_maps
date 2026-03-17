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

    def enable_classification_overlay(self, enabled=True):
        """Enable/disable the classification overlay checkbox."""
        self.classification_check.setEnabled(enabled)

    def enable_density_overlay(self, enabled=True):
        """Enable/disable the density overlay checkbox."""
        self.density_check.setEnabled(enabled)
