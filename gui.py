"""
Graphical interface for YOLO detector
"""

import sys
import cv2
import numpy as np
import time
import os
import pynvml
import pandas as pd
import logging

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSlider, QComboBox,
                             QCheckBox, QGroupBox, QTabWidget, QTextEdit, QSplitter,
                             QMessageBox, QStatusBar, QFrame, QToolTip, QProgressBar)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSize

import config
from utils import convert_cv_to_qt, resize_image, get_system_metrics
from file_manager import save_detection_results, save_txt_file, generate_unique_filename

class YOLODetectorGUI(QMainWindow):
    def __init__(self, detector):
        """
        Initialize the YOLO Detector GUI.

        Args:
            detector (YOLODetector): Instance of YOLODetector for object detection.
        """
        super().__init__()

        # Logger settings
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs.log"),
                logging.StreamHandler()
            ]
        )
        logging.info("Starting YOLO Detector GUI")

        self.setWindowIcon(QIcon('assets/icon.png'))

        # Detector initialization
        self.detector = detector

        # Variables for video stream
        self.video_source = None
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Current image and results
        self.current_image = None
        self.detection_results = []
        self.processing_time = 0

        # Variables for batch processing
        self.input_folder = None
        self.output_folder = None
        self.batch_processing_active = False

        # Initialize GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            logging.info("GPU monitoring initialized")
        except pynvml.NVMLError as e:
            self.gpu_available = False
            logging.warning(f"GPU monitoring is not available: {e}")

        # Timer for real-time system metrics updates
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_system_metrics_label)
        self.metrics_timer.start(500)  # Update every 500 ms

        # GUI setup
        self.setup_ui()

        # Try to load the model
        self.load_model()

        # Display default "waiting" image
        self.display_default_image()

    def display_default_image(self):
        """
        Display the default 'waiting' image in the image_label at startup.
        """
        waiting_image_path = os.path.join(os.path.dirname(__file__), "assets/waiting.png")
        if os.path.exists(waiting_image_path):
            pixmap = QPixmap(waiting_image_path)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            logging.info("Displayed default waiting image")
        else:
            logging.warning("Default image 'waiting.png' not found")
            self.image_label.clear()

    def setup_ui(self):
        """
        Set up the main GUI interface with a splitter for image display and control panel.
        """
        logging.info("Setting up GUI")
        # Main window parameters
        self.setWindowTitle(config.WINDOW_TITLE)
        self.resize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Create splitter to separate image and control panel
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left part - image and detection information
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        image_layout.addWidget(self.image_label)

        # System metrics label
        self.time_label = QLabel("CPU: ---% | GPU: ---% | Memory: ---%")
        self.time_label.setAlignment(Qt.AlignRight)
        self.time_label.setStyleSheet("font-weight: bold; color: #444;")
        image_layout.addWidget(self.time_label)

        # Detection results group
        detection_results_group = QGroupBox("Detection Results")
        detection_results_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        detection_results_layout = QVBoxLayout(detection_results_group)

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(150)
        self.results_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        detection_results_layout.addWidget(self.results_text)
        image_layout.addWidget(detection_results_group)

        # Adding to splitter
        splitter.addWidget(image_container)

        # Right part - control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # Tabs for different functions
        tabs = QTabWidget()
        tabs.addTab(self.create_main_tab(), "Main functions")
        tabs.addTab(self.create_batch_tab(), "Batch Processing")
        tabs.addTab(self.create_info_tab(), "About")
        control_layout.addWidget(tabs)

        # Adding to splitter
        splitter.addWidget(control_panel)

        # Setting size ratio in splitter
        splitter.setSizes([int(config.WINDOW_WIDTH * 0.7), int(config.WINDOW_WIDTH * 0.3)])

        # Creating StatusBar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Software is up-to-date")

    def create_main_tab(self):
        """
        Create the 'Main functions' tab with buttons for image/video input and detection controls.

        Returns:
            QWidget: The main tab widget.
        """
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout(main_tab)

        # Button group for image source
        source_group = QGroupBox("Image source")
        source_layout = QVBoxLayout(source_group)

        # Style for buttons
        button_style = """
            QPushButton {
                background-color: white;
                color: #333333;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                border: 2px solid #d3d3d3;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
        """

        # Buttons for main functions
        image_btn = QPushButton("📁 Upload image file")
        image_btn.setFixedHeight(50)
        image_btn.setStyleSheet(button_style)
        image_btn.clicked.connect(self.load_image)
        source_layout.addWidget(image_btn)

        video_btn = QPushButton("📁 Upload video file")
        video_btn.setFixedHeight(50)
        video_btn.setStyleSheet(button_style)
        video_btn.clicked.connect(self.load_video)
        source_layout.addWidget(video_btn)

        camera_btn = QPushButton("🎥 Start WebCam")
        camera_btn.setFixedHeight(50)
        camera_btn.setStyleSheet(button_style)
        camera_btn.clicked.connect(self.start_camera)
        source_layout.addWidget(camera_btn)

        stop_btn = QPushButton("🛑 Stop video/camera processing")
        stop_btn.setFixedHeight(50)
        stop_btn.setStyleSheet(button_style)
        stop_btn.clicked.connect(self.stop_video)
        source_layout.addWidget(stop_btn)

        main_tab_layout.addWidget(source_group)

        # Group for detection control
        detection_group = QGroupBox("Detection control")
        detection_layout = QVBoxLayout(detection_group)

        # Button for image detection
        detect_btn = QPushButton("🟢 START DETECTION")
        detect_btn.setMinimumHeight(60)
        detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #367c39;
            }
        """)
        detect_btn.clicked.connect(self.detect_objects)
        detection_layout.addWidget(detect_btn)

        # Confidence threshold settings
        conf_layout = QHBoxLayout()
        conf_label = QLabel("CW Confidence level:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(90)
        self.conf_slider.setValue(int(config.CONFIDENCE_THRESHOLD * 100))
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.update_confidence)

        self.conf_value_label = QLabel(f"{config.CONFIDENCE_THRESHOLD:.2f}")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)

        detection_layout.addLayout(conf_layout)

        # Button to save results
        save_btn = QPushButton("💾 Save result")
        save_btn.setMinimumHeight(60)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0a69b7;
            }
        """)
        save_btn.clicked.connect(self.save_result)
        detection_layout.addWidget(save_btn)

        # Button to clear image
        clear_btn = QPushButton("🗑️ Clear Image")
        clear_btn.setMinimumHeight(60)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
            QPushButton:pressed {
                background-color: #cc7a00;
            }
        """)
        clear_btn.clicked.connect(self.clear_image)
        detection_layout.addWidget(clear_btn)

        main_tab_layout.addWidget(detection_group)
        return main_tab

    def create_batch_tab(self):
        """
        Create the 'Batch Processing' tab with folder selection, processing controls, and output format selection.

        Returns:
            QWidget: The batch processing tab widget.
        """
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)

        # Folder selection group
        folder_group = QGroupBox("Folder selection")
        folder_layout = QVBoxLayout(folder_group)

        # Style for buttons
        button_style = """
            QPushButton {
                background-color: white;
                color: #333333;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                border: 2px solid #d3d3d3;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
        """

        # Buttons for folder selection
        select_input_btn = QPushButton("📁 Select Input Folder")
        select_input_btn.setFixedHeight(50)
        select_input_btn.setStyleSheet(button_style)
        select_input_btn.clicked.connect(self.select_input_folder)
        folder_layout.addWidget(select_input_btn)

        select_output_btn = QPushButton("📁 Select Output Folder")
        select_output_btn.setFixedHeight(50)
        select_output_btn.setStyleSheet(button_style)
        select_output_btn.clicked.connect(self.select_output_folder)
        folder_layout.addWidget(select_output_btn)

        batch_layout.addWidget(folder_group)

        # Processing control group
        processing_group = QGroupBox("Processing control")
        processing_layout = QVBoxLayout(processing_group)

        # Button to start batch processing
        self.process_btn = QPushButton("🟢 Start Batch Processing")
        self.process_btn.setMinimumHeight(60)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #367c39;
            }
        """)
        self.process_btn.clicked.connect(self.process_batch)
        processing_layout.addWidget(self.process_btn)

        # Confidence threshold settings for batch processing
        batch_conf_layout = QHBoxLayout()
        batch_conf_label = QLabel("CW Confidence level:")
        self.batch_conf_slider = QSlider(Qt.Horizontal)
        self.batch_conf_slider.setMinimum(10)
        self.batch_conf_slider.setMaximum(90)
        self.batch_conf_slider.setValue(int(config.CONFIDENCE_THRESHOLD * 100))
        self.batch_conf_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_conf_slider.setTickInterval(10)
        self.batch_conf_slider.valueChanged.connect(self.update_confidence)

        self.batch_conf_value_label = QLabel(f"{config.CONFIDENCE_THRESHOLD:.2f}")
        batch_conf_layout.addWidget(batch_conf_label)
        batch_conf_layout.addWidget(self.batch_conf_slider)
        batch_conf_layout.addWidget(self.batch_conf_value_label)

        processing_layout.addLayout(batch_conf_layout)

        # Checkboxes for selecting output formats
        output_format_group = QGroupBox("Output Formats")
        output_format_layout = QHBoxLayout(output_format_group)
        self.excel_checkbox = QCheckBox("Excel")
        self.excel_checkbox.setChecked(True)
        self.excel_checkbox.stateChanged.connect(
            lambda: logging.info(f"Excel checkbox changed to: {self.excel_checkbox.isChecked()}")
        )
        self.csv_checkbox = QCheckBox("CSV")
        self.csv_checkbox.setChecked(True)
        self.csv_checkbox.stateChanged.connect(
            lambda: logging.info(f"CSV checkbox changed to: {self.csv_checkbox.isChecked()}")
        )
        output_format_layout.addWidget(self.excel_checkbox)
        output_format_layout.addWidget(self.csv_checkbox)
        processing_layout.addWidget(output_format_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        processing_layout.addWidget(self.progress_bar)

        # Button to stop batch processing
        self.stop_batch_btn = QPushButton("🛑 Stop Batch Processing")
        self.stop_batch_btn.setMinimumHeight(60)
        self.stop_batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c1170a;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.stop_batch_btn.clicked.connect(self.stop_batch_processing)
        self.stop_batch_btn.setEnabled(False)
        processing_layout.addWidget(self.stop_batch_btn)

        batch_layout.addWidget(processing_group)
        return batch_tab

    def create_info_tab(self):
        """
        Create the 'About' tab with information about the application and a logo.

        Returns:
            QWidget: The about tab widget.
        """
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)

        # Software info
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setHtml("""
            <h3>About YOLO Detector</h3>
            <p>This application uses YOLO (You Only Look Once) algorithm for object detection.</p>
            <p>You can detect objects in images, videos, webcam feed, or batch process image folders.</p>
            <p>Batch processing saves images to 'images/', text files to 'txt/', and Excel/CSV results to the output folder.</p>
            <p>Developed for educational purposes.</p>
            <br>
            <p><b>Controls:</b></p>
            <ul>
                <li>Upload image/video or start webcam</li>
                <li>Adjust confidence threshold using slider</li>
                <li>Save detection results to file</li>
                <li>Batch process images in a folder (results saved to Excel/CSV)</li>
            </ul>
        """)
        info_layout.addWidget(about_text)
        return info_tab

    def load_model(self):
        """
        Load the YOLO model using the detector instance.
        """
        logging.info("Loading model...")
        if self.detector.load_model():
            logging.info(f"Model {self.detector.model_path} was loaded successfully")
            self.statusBar.showMessage(f"Model {self.detector.model_path} was loaded successfully")
        else:
            logging.error("Model loading error")
            self.statusBar.showMessage("Model loading error")
            QMessageBox.critical(self, "Error", "Failed to load YOLO model")

    def load_image(self):
        """
        Load an image from a file and display it in the GUI.
        """
        self.stop_video()
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Open image", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if image_path:
            try:
                self.current_image = cv2.imread(image_path)
                if self.current_image is None:
                    raise Exception("Failed to open image")
                self.display_image(self.current_image)
                logging.info(f"Image loaded: {image_path}")
                self.statusBar.showMessage(f"Image loaded: {image_path}")
            except Exception as e:
                logging.error(f"Error loading image: {e}")
                self.statusBar.showMessage(f"Error loading image: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load image: {e}")
                self.display_default_image()  # Restore default image on error

    def load_video(self):
        """
        Load a video from a file and start playback.
        """
        self.stop_video()
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self, "Open video", "", "Video (*.mp4 *.avi *.mov *.mkv)"
        )
        if video_path:
            try:
                self.video_capture = cv2.VideoCapture(video_path)
                if not self.video_capture.isOpened():
                    raise Exception("Failed to open video")
                self.timer.start(30)
                self.video_source = video_path
                logging.info(f"Playing video: {video_path}")
                self.statusBar.showMessage(f"Playing video: {video_path}")
            except Exception as e:
                logging.error(f"Error loading video: {e}")
                self.statusBar.showMessage(f"Error loading video: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load video: {e}")
                self.display_default_image()  # Restore default image on error

    def start_camera(self):
        """
        Start the webcam feed.
        """
        self.stop_video()
        try:
            self.video_capture = cv2.VideoCapture(config.DEFAULT_VIDEO_SOURCE)
            if not self.video_capture.isOpened():
                raise Exception("Failed to open camera")
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
            self.timer.start(30)
            self.video_source = config.DEFAULT_VIDEO_SOURCE
            logging.info("Playing video from camera")
            self.statusBar.showMessage("Playing video from camera")
        except Exception as e:
            logging.error(f"Error starting camera: {e}")
            self.statusBar.showMessage(f"Error starting camera: {e}")
            QMessageBox.warning(self, "Error", f"Failed to start camera: {e}")
            self.display_default_image()  # Restore default image on error

    def stop_video(self):
        """
        Stop video or webcam playback.
        """
        if self.timer.isActive():
            self.timer.stop()
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
            self.video_capture = None
            self.video_source = None
            logging.info("Video playback stopped")
            self.statusBar.showMessage("Video playback stopped")
        self.display_default_image()  # Restore default image

    def update_frame(self):
        """
        Update the video frame from the current video source.
        """
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                if self.video_source == config.DEFAULT_VIDEO_SOURCE:
                    frame = cv2.flip(frame, 1)
                self.current_image = frame.copy()
                if self.detector.is_ready:
                    start_time = time.time()
                    frame, self.detection_results = self.detector.detect(frame)
                    self.processing_time = time.time() - start_time
                    self.update_results_info()
                self.display_image(frame)
            else:
                if isinstance(self.video_source, str):
                    self.stop_video()
                    logging.info("Video playback finished")
                    self.statusBar.showMessage("Video playback finished")

    def detect_objects(self):
        """
        Perform object detection on the current image.
        """
        if self.current_image is None:
            logging.warning("No image or video selected for detection")
            QMessageBox.warning(self, "Warning",
                                "No image or video selected. Please load an image, video, or start the webcam.")
            self.statusBar.showMessage("No image for detection")
            return
        if not self.detector.is_ready:
            logging.warning("Model not loaded")
            self.statusBar.showMessage("Model not loaded")
            return
        try:
            logging.info("Performing object detection")
            self.statusBar.showMessage("Performing object detection...")
            start_time = time.time()
            result_image, self.detection_results = self.detector.detect(self.current_image)
            self.processing_time = time.time() - start_time
            self.display_image(result_image)
            self.update_results_info()
            logging.info(f"Detection completed. Objects found: {len(self.detection_results)}")
            self.statusBar.showMessage(f"Detection completed. Objects found: {len(self.detection_results)}")
        except Exception as e:
            logging.error(f"Detection error: {e}")
            self.statusBar.showMessage(f"Detection error: {e}")
            QMessageBox.warning(self, "Error", f"Error during object detection: {e}")
            self.display_default_image()  # Restore default image on error

    def update_system_metrics_label(self):
        """
        Update the label displaying CPU, GPU, and memory usage.
        """
        cpu_usage, gpu_usage, memory_usage = get_system_metrics(self.gpu_available)
        label_text = f"CPU: {cpu_usage:.1f}% | GPU: {gpu_usage}% | Memory: {memory_usage:.1f}%"
        self.time_label.setText(label_text)

    def update_results_info(self):
        """
        Update the detection results text area with detection information.
        """
        self.results_text.clear()
        if not self.detection_results:
            self.results_text.setText("No objects found")
            return

        class_counts = {}
        for detection in self.detection_results:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            if class_name in class_counts:
                class_counts[class_name].append(confidence)
            else:
                class_counts[class_name] = [confidence]

        result_text = f"Total objects found: {len(self.detection_results)}\n"
        result_text += f"Processing time: {self.processing_time:.3f} seconds\n\n"
        for class_name, confidences in class_counts.items():
            count = len(confidences)
            avg_conf = sum(confidences) / count
            result_text += f"- {class_name}: {count} (avg. confidence: {avg_conf:.2f})\n"
        result_text += "\nDetailed information:\n"
        for i, detection in enumerate(self.detection_results):
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            confidence_percent = detection["confidence_percent"]
            x1, y1, x2, y2 = detection["box"]
            result_text += f"{i + 1}. {class_name} ({confidence_percent:.1f}%)\n"
            result_text += f"   Position: ({x1}, {y1}) - ({x2}, {y2})\n"
        self.results_text.setText(result_text)

    def update_confidence(self):
        """
        Update the confidence threshold for the detector and synchronize both sliders.
        """
        sender = self.sender()
        if sender == self.batch_conf_slider:
            value = self.batch_conf_slider.value() / 100.0
            self.conf_slider.setValue(int(value * 100))
        else:
            value = self.conf_slider.value() / 100.0
            self.batch_conf_slider.setValue(int(value * 100))

        self.conf_value_label.setText(f"{value:.2f}")
        self.batch_conf_value_label.setText(f"{value:.2f}")
        self.detector.confidence_threshold = value
        logging.info(f"Confidence threshold set to {value:.2f}")
        self.statusBar.showMessage(f"Confidence threshold set to {value:.2f}")

    def display_image(self, image):
        """
        Display an image in the GUI.

        Args:
            image (np.ndarray): Image to display (BGR format).
        """
        if image is None:
            self.display_default_image()
            return
        resized_image = resize_image(image, self.image_label.width(), self.image_label.height())
        q_image = convert_cv_to_qt(resized_image)
        if q_image:
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

    def clear_image(self):
        """
        Clear the current image and related data from the GUI.
        """
        self.image_label.clear()
        self.current_image = None
        self.detection_results = []
        self.results_text.clear()
        self.results_text.setText("Image cleared")
        logging.info("Image cleared")
        self.statusBar.showMessage("Image cleared")
        self.display_default_image()  # Restore default image

    def save_result(self):
        """
        Save the detection result to images/ and txt/ subfolders with an auto-generated filename.
        """
        if self.current_image is None:
            logging.warning("No image to save")
            self.statusBar.showMessage("No image to save")
            return
        if self.detector.is_ready:
            start_time = time.time()
            result_image, detection_results = self.detector.detect(self.current_image)
            processing_time = time.time() - start_time
        else:
            result_image = self.current_image
            detection_results = []
            processing_time = 0

        file_dialog = QFileDialog()
        save_dir = file_dialog.getExistingDirectory(self, "Select folder to save result")
        if save_dir:
            # Create subfolders
            images_folder = os.path.join(save_dir, "images")
            txt_folder = os.path.join(save_dir, "txt")
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(txt_folder, exist_ok=True)

            # Generate unique filename
            image_filename = generate_unique_filename(prefix="detection", extension=".jpg")
            image_path = os.path.join(images_folder, image_filename)

            # Save image
            if save_detection_results(result_image, image_path):
                # Save text file with detection details
                txt_filename = os.path.splitext(image_filename)[0] + ".txt"
                txt_path = os.path.join(txt_folder, txt_filename)
                if save_txt_file(txt_path, image_filename, processing_time, detection_results):
                    logging.info(f"Result and details saved to {image_path} and {txt_path}")
                    self.statusBar.showMessage(f"Result and details saved to {image_path} and {txt_path}")
                else:
                    logging.error("Error saving text file")
                    self.statusBar.showMessage("Error saving text file")
                    QMessageBox.warning(self, "Error", "Failed to save text file")
            else:
                logging.error("Error saving image")
                self.statusBar.showMessage("Error saving image")
                QMessageBox.warning(self, "Error", "Failed to save image")

    def select_input_folder(self):
        """
        Select the input folder for batch processing.
        """
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(self, "Select Input Folder")
        if folder_path:
            self.input_folder = folder_path
            logging.info(f"Input folder selected: {folder_path}")
            self.statusBar.showMessage(f"Input folder selected: {folder_path}")
            self.results_text.append(f"Input folder: {folder_path}")

    def select_output_folder(self):
        """
        Select the output folder for batch processing results.
        """
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_folder = folder_path
            logging.info(f"Output folder selected: {folder_path}")
            self.statusBar.showMessage(f"Output folder selected: {folder_path}")
            self.results_text.append(f"Output folder: {folder_path}")

    def collect_batch_result(self, filename, detection_results, processing_time):
        """
        Collect detection results for a single image for Excel output.

        Args:
            filename (str): Name of the processed image file.
            detection_results (list): List of detection dictionaries.
            processing_time (float): Time taken for detection in seconds.

        Returns:
            dict: Dictionary with Excel row data.
        """
        class_counts = {}
        for detection in detection_results:
            class_name = detection["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        classes_detected = ", ".join([f"{k}: {v}" for k, v in class_counts.items()]) or "None"
        return {
            "Filename": filename,
            "Objects Found": len(detection_results),
            "Processing Time (s)": round(processing_time, 3),
            "Classes Detected": classes_detected
        }

    def process_batch(self):
        """
        Process all images in the input folder and save results to the output folder.
        Images are saved to 'images/', text files to 'txt/', and Excel/CSV files to the root of output_folder based on selected formats.
        Displays a "loading" image during processing.
        """
        if not hasattr(self, 'input_folder') or not self.input_folder:
            logging.warning("No input folder selected")
            self.statusBar.showMessage("No input folder selected")
            QMessageBox.warning(self, "Error", "Please select an input folder")
            return
        if not hasattr(self, 'output_folder') or not self.output_folder:
            logging.warning("No output folder selected")
            self.statusBar.showMessage("No output folder selected")
            QMessageBox.warning(self, "Error", "Please select an output folder")
            return
        if not self.detector.is_ready:
            logging.warning("Model not loaded")
            self.statusBar.showMessage("Model not loaded")
            QMessageBox.warning(self, "Error", "Model not loaded")
            return
        if not (self.excel_checkbox.isChecked() or self.csv_checkbox.isChecked()):
            logging.warning("No output format selected")
            self.statusBar.showMessage("No output format selected")
            QMessageBox.warning(self, "Error", "Please select at least one output format (Excel or CSV)")
            return

        # Display "loading" image
        loading_image_path = os.path.join(os.path.dirname(__file__), "assets/loading.png")
        if os.path.exists(loading_image_path):
            pixmap = QPixmap(loading_image_path)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            logging.info("Displayed loading image during batch processing")
        else:
            logging.warning("Loading image 'loading.png' not found")
            self.image_label.clear()

        # Reset progress bar
        self.progress_bar.setValue(0)

        self.batch_processing_active = True
        self.process_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        self.results_text.clear()
        logging.info("Starting batch processing")
        self.results_text.append("Starting batch processing...")

        # Debug: showing checkbox states in GUI
        self.results_text.append(
            f"Selected formats: Excel={self.excel_checkbox.isChecked()}, CSV={self.csv_checkbox.isChecked()}")

        # Creating subfolders for results
        images_folder = os.path.join(self.output_folder, "images")
        txt_folder = os.path.join(self.output_folder, "txt")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(txt_folder, exist_ok=True)

        # Image processing
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        processed_count = 0
        error_count = 0
        batch_results = []
        total_files = len([f for f in os.listdir(self.input_folder) if f.lower().endswith(image_extensions)])
        self.progress_bar.setMaximum(total_files)

        try:
            for filename in os.listdir(self.input_folder):
                if not self.batch_processing_active:
                    logging.info("Batch processing stopped by user")
                    self.results_text.append("Batch processing stopped by user")
                    self.statusBar.showMessage("Batch processing stopped")
                    break
                if filename.lower().endswith(image_extensions):
                    image_path = os.path.join(self.input_folder, filename)
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            raise Exception("Failed to load image")
                        start_time = time.time()
                        result_image, detection_results = self.detector.detect(image)
                        processing_time = time.time() - start_time
                        output_image_path = os.path.join(images_folder, f"detected_{filename}")
                        if not save_detection_results(result_image, output_image_path):
                            raise Exception("Failed to save result image")
                        output_txt_path = os.path.join(txt_folder, f"detected_{filename.rsplit('.', 1)[0]}.txt")
                        if not save_txt_file(output_txt_path, filename, processing_time, detection_results):
                            raise Exception("Failed to save text file")

                        # Data collection for Excel and CSV
                        batch_results.append(self.collect_batch_result(filename, detection_results, processing_time))

                        logging.info(f"Processed {filename}: {len(detection_results)} objects found")
                        self.results_text.append(f"Processed {filename}: {len(detection_results)} objects found")
                        processed_count += 1
                        self.progress_bar.setValue(processed_count)
                        QApplication.processEvents()
                    except Exception as e:
                        logging.error(f"Error processing {filename}: {str(e)}")
                        self.results_text.append(f"Error processing {filename}: {str(e)}")
                        error_count += 1
                        continue

            if self.batch_processing_active:
                logging.info(f"Batch processing completed: {processed_count} images processed, {error_count} errors")
                self.results_text.append(
                    f"\nBatch processing completed: {processed_count} images processed, {error_count} errors")
                self.statusBar.showMessage(
                    f"Batch processing completed: {processed_count} images processed, {error_count} errors")

                # Saving results in selected formats
                if batch_results:
                    saved_files = []
                    try:
                        df = pd.DataFrame(batch_results)
                        logging.info(f"Excel checkbox state: {self.excel_checkbox.isChecked()}")
                        logging.info(f"CSV checkbox state: {self.csv_checkbox.isChecked()}")

                        if self.excel_checkbox.isChecked():
                            excel_filename = generate_unique_filename(prefix="batch_results", extension=".xlsx")
                            excel_path = os.path.join(self.output_folder, excel_filename)
                            df.to_excel(excel_path, index=False)
                            saved_files.append(excel_path)
                            logging.info(f"Saved Excel file: {excel_path}")
                        else:
                            logging.info("Excel output skipped (checkbox unchecked)")

                        if self.csv_checkbox.isChecked():
                            csv_filename = generate_unique_filename(prefix="batch_results", extension=".csv")
                            csv_path = os.path.join(self.output_folder, csv_filename)
                            df.to_csv(csv_path, index=False)
                            saved_files.append(csv_path)
                            logging.info(f"Saved CSV file: {csv_path}")
                        else:
                            logging.info("CSV output skipped (checkbox unchecked)")

                        if saved_files:
                            saved_files_str = " and ".join(saved_files)
                            logging.info(f"Batch results saved to {saved_files_str}")
                            self.results_text.append(f"Batch results saved to {saved_files_str}")
                            self.statusBar.showMessage(f"Batch results saved to {saved_files_str}")
                        else:
                            logging.warning("No files saved (no formats selected)")
                            self.results_text.append("No files saved (no formats selected)")
                            self.statusBar.showMessage("No files saved")
                    except Exception as e:
                        logging.error(f"Error saving results: {str(e)}")
                        self.results_text.append(f"Error saving results: {str(e)}")
                        self.statusBar.showMessage(f"Error saving results: {e}")
                        QMessageBox.warning(self, "Error", f"Failed to save results: {e}")

        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            self.results_text.append(f"Batch processing error: {str(e)}")
            self.statusBar.showMessage(f"Batch processing error: {e}")
            QMessageBox.warning(self, "Error", f"Error during batch processing: {e}")
        finally:
            self.batch_processing_active = False
            self.process_btn.setEnabled(True)
            self.stop_batch_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.display_default_image()  # Restore default image
            logging.info("Restored default waiting image after batch processing")

    def stop_batch_processing(self):
        """
        Stop the batch processing operation.
        """
        self.batch_processing_active = False
        self.process_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.display_default_image()  # Restore default image
        logging.info("Batch processing stopped and restored default waiting image")

    def closeEvent(self, event):
        """
        Handle the window close event.

        Args:
            event: The close event.
        """
        self.stop_video()
        self.metrics_timer.stop()
        if self.gpu_available:
            pynvml.nvmlShutdown()
        logging.info("Application closed")
        super().closeEvent(event)