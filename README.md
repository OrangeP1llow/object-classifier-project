YOLO Object Detection
A GUI-based application for object detection using YOLOv8. Supports processing of images, videos, webcam feeds, and batch operations with customizable output in Excel and CSV formats.
Features

Real-time object detection in images, videos, or webcam feeds using YOLOv8.
Batch processing of images with results saved to images/, txt/, and user-selected Excel and/or CSV files.
Real-time monitoring of CPU, GPU, and memory usage.
Adjustable confidence threshold via synchronized sliders and spinbox.
Save individual detection results as images and text files in images/ and txt/ subfolders.
Progress bar for batch processing to track operation progress.
Comprehensive event logging to logs.log for debugging and monitoring.
Custom application icon and default images:
waiting.png: Displayed at startup and after clearing/resetting the image display.
loading.png: Shown during batch processing for enhanced user experience.


Batch processing tab with checkboxes to select output formats (Excel, CSV, or both). At least one format must be selected, or an error message is displayed.
Structured project with modular code and English documentation.

Installation

Install Python 3.8+ from python.org.
Clone the repository:git clone https://github.com/OrangeP1llow/object-classifier-project.git
cd object-classifier-project


Install dependencies:pip install -r requirements.txt


Download the YOLOv8 model (yolov8x.pt) from Google Drive and place it in the project root.
Ensure the assets/ folder contains waiting.png, loading.png, and icon.png.

Usage
Run the application:
python main.py

Interface

Main Functions Tab:
Upload an image or video, or start the webcam feed.
Adjust the confidence threshold and initiate detection.
Save results to images/ and txt/ subfolders.


Batch Processing Tab:
Select input and output folders.
Choose output formats (Excel, CSV, or both) using checkboxes.
Process all images and save results to images/, txt/, and selected Excel/CSV files.
Displays loading.png during processing and waiting.png by default or after completion/cancellation.
Monitor progress with a progress bar.


About Tab:
Provides information about the application.



Logs

All operations (model loading, image processing, errors, checkbox states, and image displays) are logged to logs.log in the project root.
View logs:cat logs.log  # On Linux/Mac
type logs.log  # On Windows



Branching Strategy

main: Contains only the README.md for project overview.
develop: Full project codebase, including source files, assets, and documentation.

Screenshots

Batch Processing Tab (showing loading.png and progress bar):
Default View (showing waiting.png):

Output Structure

Single Detection:
Images: save_folder/images/detection_*.jpg
Text files: save_folder/txt/detection_*.txt


Batch Processing:
Images: output_folder/images/detected_*.jpg
Text files: output_folder/txt/detected_*.txt
Excel (if selected): output_folder/batch_results_*.xlsx (columns: Filename, Objects Found, Processing Time (s), Classes Detected)
CSV (if selected): output_folder/batch_results_*.csv (same columns as Excel)



Project Structure
object-classifier-project/
├── assets/
│   ├── waiting.png        # Default image at startup and after reset
│   ├── loading.png        # Displayed during batch processing
│   └── icon.png           # Application icon
├── docs/
│   ├── default_view.png   # Screenshot of default GUI view
│   └── batch_processing.png  # Screenshot of batch processing
├── main.py                # Application entry point
├── gui.py                 # GUI implementation using PyQt5
├── detector.py            # YOLOv8 detection logic
├── utils.py               # Utilities for image processing and system monitoring
├── file_manager.py        # File saving and naming utilities
├── config.py              # Configuration settings
├── requirements.txt       # Project dependencies
├── .gitignore             # Excludes logs and temporary files
└── README.md              # Project documentation

Requirements
Dependencies are listed in requirements.txt. Key libraries include:

PyQt5
opencv-python
numpy
pandas
ultralytics
psutil
pynvml
torch

Install them using:
pip install -r requirements.txt

Notes

The YOLOv8 model (yolov8x.pt) is not included in the repository due to its size (~100 MB). Download it from the provided Google Drive link.
Ensure the assets/ folder is present with all required images to avoid warnings in logs.log (e.g., WARNING - Default image 'waiting.png' not found).
Logs are saved to logs.log and include detailed information for debugging.

License
This project is developed for educational purposes and is not licensed for commercial use.
Contributing
Contributions are welcome! Please create a feature branch from develop and submit a Pull Request for review.

Developed by OrangeP1llow
