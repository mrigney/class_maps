"""Dialogs: export, profile save/load."""

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QCheckBox, QGroupBox, QMessageBox,
)
from PyQt5.QtCore import Qt


class ExportDialog(QDialog):
    """Dialog for exporting classified and density rasters."""

    def __init__(self, parent=None, has_density=False, has_crs=False):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(450)

        self._output_dir = ""
        self._prefix = "classified"
        self._export_classified = True
        self._export_density = has_density
        self._export_legend = True

        layout = QVBoxLayout(self)

        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output directory:"))
        self._dir_edit = QLineEdit()
        dir_layout.addWidget(self._dir_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)

        # Filename prefix
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("Filename prefix:"))
        self._prefix_edit = QLineEdit("classified")
        prefix_layout.addWidget(self._prefix_edit)
        layout.addLayout(prefix_layout)

        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout()

        self._classified_check = QCheckBox("Classified raster (GeoTIFF)")
        self._classified_check.setChecked(True)
        options_layout.addWidget(self._classified_check)

        self._density_check = QCheckBox("Density raster (GeoTIFF)")
        self._density_check.setChecked(has_density)
        self._density_check.setEnabled(has_density)
        options_layout.addWidget(self._density_check)

        self._legend_check = QCheckBox("Class legend (JSON)")
        self._legend_check.setChecked(True)
        options_layout.addWidget(self._legend_check)

        if has_crs:
            options_layout.addWidget(QLabel("CRS information will be preserved."))
        else:
            options_layout.addWidget(QLabel("No CRS information (PNG source)."))

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._accept)
        btn_layout.addWidget(export_btn)
        layout.addLayout(btn_layout)

    def _browse_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self._dir_edit.setText(dir_path)

    def _accept(self):
        output_dir = self._dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Export", "Please select an output directory.")
            return
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Export", "Output directory does not exist.")
            return

        self._output_dir = output_dir
        self._prefix = self._prefix_edit.text().strip() or "classified"
        self._export_classified = self._classified_check.isChecked()
        self._export_density = self._density_check.isChecked()
        self._export_legend = self._legend_check.isChecked()
        self.accept()

    def get_export_settings(self):
        """Return export settings as a dict."""
        return {
            "output_dir": self._output_dir,
            "prefix": self._prefix,
            "export_classified": self._export_classified,
            "export_density": self._export_density,
            "export_legend": self._export_legend,
        }
