"""Entry point for the class_maps application."""

import sys
from PyQt5.QtWidgets import QApplication

from class_maps.gui.main_window import ClassMapsWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("class_maps")
    app.setOrganizationName("class_maps")

    window = ClassMapsWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
