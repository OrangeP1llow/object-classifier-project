import cv2
import numpy as np
import psutil
import pynvml
from PyQt5.QtGui import QImage

def convert_cv_to_qt(image):
    """
    Convert an OpenCV image to Qt format.

    Args:
        image (np.ndarray): Input image (BGR or grayscale).

    Returns:
        QImage: Converted Qt image, or None if conversion fails.
    """
    if image is None:
        return None
    height, width = image.shape[:2]
    if len(image.shape) == 2:
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:
        bytes_per_line = 3 * width
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return q_image

def resize_image(image, max_width, max_height):
    """
    Resize an image while preserving aspect ratio.

    Args:
        image (np.ndarray): Input image.
        max_width (int): Maximum width.
        max_height (int): Maximum height.

    Returns:
        np.ndarray: Resized image, or None if input is invalid.
    """
    if image is None:
        return None
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def get_system_metrics(gpu_available=True):
    """
    Get CPU, GPU, and memory usage metrics.

    Args:
        gpu_available (bool): Whether GPU monitoring is available.

    Returns:
        tuple: (cpu_usage, gpu_usage, memory_usage) as percentages.
    """
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    gpu_usage = 0
    if gpu_available:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage = util.gpu
        except pynvml.NVMLError:
            gpu_usage = 0
    return cpu_usage, gpu_usage, memory_usage