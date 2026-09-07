"""GUI startup and last-resort reporting for uncaught Qt callbacks."""
import logging
import sys
import traceback


def report_exception(exc_type, error, tb):
    logging.getLogger("src.cocoon_screening").error(
        "Unhandled application error", exc_info=(exc_type, error, tb)
    )
    message = f"{exc_type.__name__}: {error}\n\nPlease retry or restart the application."
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is None:
            raise RuntimeError("Qt is not initialized")
        QMessageBox.critical(None, "Cocoon Annotation Tool - Error", message)
    except Exception:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "Cocoon Annotation Tool", 0x10)
        elif sys.stderr is not None:
            traceback.print_exception(exc_type, error, tb)


def main() -> int:
    previous_hook = sys.excepthook
    sys.excepthook = report_exception
    try:
        from PySide6.QtWidgets import QApplication
        qt_app = QApplication.instance() or QApplication(sys.argv)
        qt_app.setStyle("Fusion")
        qt_app.setApplicationName("Cocoon Annotation Tool")

        from src.cocoon_screening.io.logging import setup_logging
        from src.cocoon_screening.paths import user_log_path
        setup_logging(user_log_path())

        from src.cocoon_screening.ui.main_window import MainWindow
        window = MainWindow()
        window.show()
        return qt_app.exec()
    except Exception:
        report_exception(*sys.exc_info())
        return 1
    finally:
        sys.excepthook = previous_hook
