from PySide6.QtWidgets import QApplication
from src.cocoon_screening.ui.main_window import MainWindow
from src.cocoon_screening.io.logging import setup_logging
from src.cocoon_screening.paths import user_log_path
import os

basedir = os.path.dirname(__file__)

def main() -> None:
    setup_logging(user_log_path())

    qt_app = QApplication([])
    qt_app.setStyle("Fusion")
    qt_app.setApplicationName("Cocoon Annotation Tool")

    window = MainWindow()
    window.show()

    qt_app.exec()