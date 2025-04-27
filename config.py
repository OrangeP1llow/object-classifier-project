"""
Configuration settings for the YOLO Object Detection project.
Contains paths, GUI settings, detector parameters, and color schemes.
"""

# File path
MODEL_PATH = "yolov8x.pt"  # Possible var-s: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Interface settings
WINDOW_TITLE = "Object Classifier"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# Detector settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Color settings (BGR)
COLORS = {
    "bbox": (0, 255, 0),
    "text_bg": (0, 0, 0),
    "text": (255, 255, 255)
}

# Video settings
DEFAULT_VIDEO_SOURCE = 0  # 0 - IT'S A WEBCAM!
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480