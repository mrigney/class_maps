"""Class palette sidebar: class list with add/remove/rename/color editing."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QColorDialog, QInputDialog, QLabel, QGroupBox,
    QMessageBox,
)
from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal

from class_maps.config import DEFAULT_CLASSES
from class_maps.utils.color_utils import get_next_class_id, generate_distinct_color


class ClassPalette(QWidget):
    """Sidebar widget for managing landcover classes.

    Signals
    -------
    active_class_changed(int)
        Emitted when the user selects a different class. Argument is class_id.
    classes_changed()
        Emitted when classes are added, removed, or modified.
    """

    active_class_changed = pyqtSignal(int)
    classes_changed = pyqtSignal()

    def __init__(self, class_definitions=None, parent=None):
        super().__init__(parent)
        if class_definitions is None:
            class_definitions = dict(DEFAULT_CLASSES)
        self._class_defs = class_definitions
        self._active_class_id = None
        self._labeled_counts = {}  # {class_id: count}
        self._init_ui()
        self._refresh_list()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        group = QGroupBox("Landcover Classes")
        group_layout = QVBoxLayout()

        # Class list
        self._list_widget = QListWidget()
        self._list_widget.currentRowChanged.connect(self._on_selection_changed)
        self._list_widget.itemDoubleClicked.connect(self._rename_class)
        group_layout.addWidget(self._list_widget)

        # Buttons row 1
        btn_row1 = QHBoxLayout()
        self._add_btn = QPushButton("Add")
        self._add_btn.clicked.connect(self._add_class)
        btn_row1.addWidget(self._add_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_class)
        btn_row1.addWidget(self._remove_btn)

        self._color_btn = QPushButton("Color")
        self._color_btn.clicked.connect(self._edit_color)
        btn_row1.addWidget(self._color_btn)
        group_layout.addLayout(btn_row1)

        # Active class label
        self._active_label = QLabel("Active: (none)")
        group_layout.addWidget(self._active_label)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _refresh_list(self):
        """Rebuild the list widget from class definitions."""
        current_row = self._list_widget.currentRow()
        self._list_widget.clear()

        for class_id, cdef in self._class_defs.items():
            color = cdef["color"]
            name = cdef["name"]
            count = self._labeled_counts.get(class_id, 0)

            # Create color swatch icon
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(*color))
            icon = QIcon(pixmap)

            text = f"{name} ({count})" if count > 0 else name
            item = QListWidgetItem(icon, text)
            item.setData(Qt.UserRole, class_id)
            self._list_widget.addItem(item)

        # Restore selection
        if 0 <= current_row < self._list_widget.count():
            self._list_widget.setCurrentRow(current_row)
        elif self._list_widget.count() > 0:
            self._list_widget.setCurrentRow(0)

    def _on_selection_changed(self, row):
        """Handle class selection change."""
        if row < 0:
            self._active_class_id = None
            self._active_label.setText("Active: (none)")
            return

        item = self._list_widget.item(row)
        class_id = item.data(Qt.UserRole)
        self._active_class_id = class_id
        name = self._class_defs[class_id]["name"]
        self._active_label.setText(f"Active: {name}")
        self.active_class_changed.emit(class_id)

    def _add_class(self):
        """Add a new class."""
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        if not ok or not name.strip():
            return

        new_id = get_next_class_id(self._class_defs)
        existing_colors = [c["color"] for c in self._class_defs.values()]
        color = generate_distinct_color(existing_colors)

        self._class_defs[new_id] = {"name": name.strip(), "color": color}
        self._refresh_list()
        self._list_widget.setCurrentRow(self._list_widget.count() - 1)
        self.classes_changed.emit()

    def _remove_class(self):
        """Remove the selected class."""
        row = self._list_widget.currentRow()
        if row < 0:
            return

        item = self._list_widget.item(row)
        class_id = item.data(Qt.UserRole)
        name = self._class_defs[class_id]["name"]

        reply = QMessageBox.question(
            self, "Remove Class",
            f"Remove class '{name}'? Any labels using this class will be cleared.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        del self._class_defs[class_id]
        self._labeled_counts.pop(class_id, None)
        self._refresh_list()
        self.classes_changed.emit()

    def _rename_class(self, item):
        """Rename a class by double-clicking."""
        class_id = item.data(Qt.UserRole)
        old_name = self._class_defs[class_id]["name"]

        name, ok = QInputDialog.getText(
            self, "Rename Class", "New name:", text=old_name
        )
        if not ok or not name.strip():
            return

        self._class_defs[class_id]["name"] = name.strip()
        self._refresh_list()
        self.classes_changed.emit()

    def _edit_color(self):
        """Edit the color of the selected class."""
        row = self._list_widget.currentRow()
        if row < 0:
            return

        item = self._list_widget.item(row)
        class_id = item.data(Qt.UserRole)
        old_color = self._class_defs[class_id]["color"]

        color = QColorDialog.getColor(
            QColor(*old_color), self, "Select Class Color"
        )
        if not color.isValid():
            return

        self._class_defs[class_id]["color"] = (
            color.red(), color.green(), color.blue()
        )
        self._refresh_list()
        self.classes_changed.emit()

    def get_active_class_id(self):
        """Return the currently selected class ID, or None."""
        return self._active_class_id

    def get_class_definitions(self):
        """Return the current class definitions dict."""
        return self._class_defs

    def set_class_definitions(self, class_defs):
        """Replace class definitions (e.g., from a loaded profile)."""
        self._class_defs = dict(class_defs)
        self._labeled_counts = {}
        self._refresh_list()

    def update_labeled_counts(self, labeled_segments):
        """Update the count display for each class.

        Parameters
        ----------
        labeled_segments : dict
            {segment_id: class_id}
        """
        self._labeled_counts = {}
        for class_id in labeled_segments.values():
            self._labeled_counts[class_id] = self._labeled_counts.get(class_id, 0) + 1
        self._refresh_list()
