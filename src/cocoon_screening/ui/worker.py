"""Execute one serialized operation away from the GUI thread."""
import logging
from PySide6.QtCore import QObject, Signal, Slot


class Worker(QObject):
    finished = Signal(object, object)

    @Slot(object)
    def run(self, operation):
        try:
            result = operation()
        except Exception as error:
            logging.getLogger(__name__).exception("Background operation failed")
            self.finished.emit(None, error)
        else:
            self.finished.emit(result, None)
