"""
Module for working with the YOLO model
"""

import cv2
import numpy as np
import torch
import os
import sys
import logging
from ultralytics import YOLO
import config

def resource_path(relative_path):
    """Отримує правильний шлях до ресурсу, працює як у розробці, так і в .exe"""
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(".")
    full_path = os.path.join(base_path, relative_path)
    logging.info(f"Resolved resource path for {relative_path}: {full_path}")
    return full_path

class YOLODetector:
    def __init__(self, model_path=None):
        """
        Initialize the YOLO detector.

        Args:
            model_path (str, optional): Path to the YOLO model file. Defaults to config.MODEL_PATH.
        """
        self.model_path = model_path if model_path else config.MODEL_PATH
        self.model_path = resource_path(self.model_path)  # Оновлюємо шлях через resource_path
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.iou_threshold = config.IOU_THRESHOLD
        self.model = None
        self.classes = None
        self.is_ready = False
        logging.info(f"Initialized with model path: {self.model_path}")

    def load_model(self):
        """
        Load the YOLO model.

        Returns:
            bool: True if the model loaded successfully, False otherwise.
        """
        logging.info(f"Attempting to load model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            logging.error(f"Model file {self.model_path} not found. Checking directory contents...")
            try:
                dir_contents = os.listdir(os.path.dirname(self.model_path) or ".")
                logging.info(f"Directory contents: {dir_contents}")
            except Exception as e:
                logging.error(f"Failed to list directory: {e}")
            self.is_ready = False
            return False
        try:
            self.model = YOLO(self.model_path)
            self.classes = self.model.names
            self.is_ready = True
            logging.info(f"Model {self.model_path} loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            self.is_ready = False
            return False

    def detect(self, image, draw=True):
        """
        Detect objects in an image using the YOLO model.

        Args:
            image (np.ndarray): Input image in BGR format.
            draw (bool): Whether to draw bounding boxes on the image (default: True).

        Returns:
            tuple: (output_image, detections)
                - output_image (np.ndarray): Image with bounding boxes (if draw=True) or original image.
                - detections (list): List of dictionaries, each containing:
                    - class_id (int): Class ID.
                    - class_name (str): Class name.
                    - confidence (float): Confidence score (0 to 1).
                    - confidence_percent (float): Confidence score in percent (0 to 100).
                    - box (tuple): Bounding box coordinates (x1, y1, x2, y2).
        """
        if not self.is_ready or self.model is None:
            logging.warning("Model not loaded")
            return image, []
        if image is None or not isinstance(image, np.ndarray):
            logging.error("Invalid image provided")
            return image, []

        try:
            # Copy image for drawing
            output_img = image.copy() if draw else None

            # Detection performing
            results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)

            detections = []

            # Processing results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Getting coordinates, confidence, and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    confidence_percent = confidence * 100
                    class_id = int(box.cls[0])
                    class_name = self.classes[class_id]

                    # Add to the list of results
                    detection = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "confidence_percent": confidence_percent,
                        "box": (x1, y1, x2, y2)
                    }
                    detections.append(detection)

                    # Draw the results on the image
                    if draw:
                        self._draw_detection(output_img, detection)

            logging.info(f"Detected {len(detections)} objects")
            return output_img if draw else image, detections

        except Exception as e:
            logging.error(f"Detection error: {e}")
            return image, []

    def _draw_detection(self, image, detection):
        """
        Draw a bounding box and label on the image.

        Args:
            image (np.ndarray): Image to draw on.
            detection (dict): Detection dictionary with box, class_name, and confidence.
        """
        x1, y1, x2, y2 = detection["box"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        # Draw a frame
        cv2.rectangle(image, (x1, y1), (x2, y2), config.COLORS["bbox"], 2)

        # Text preparation
        label = f"{class_name}: {confidence:.2f}"

        # Text sizes
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_w, text_h = text_size

        # Drawing a background for the text
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), config.COLORS["text_bg"], -1)

        # Drawing text
        cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLORS["text"], 2)