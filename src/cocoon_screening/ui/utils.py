import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

def array_to_pixmap(image):
    if image.ndim == 2:
        display_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        display_image = display_image.astype(np.uint8)
        height, width = display_image.shape
        qimage = QImage(display_image.data, width, height, width, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage.copy())

    if image.shape[2] == 4:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        height, width, channels = display_image.shape
        qimage = QImage(display_image.data, width, height, channels * width, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimage.copy())

    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = display_image.shape
    qimage = QImage(display_image.data, width, height, channels * width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())
