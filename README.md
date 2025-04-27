# object-classifier-project

YOLO Object Detection
A GUI-based application for object detection using YOLOv8. This project supports processing images, videos, webcam feeds, and batch operations with customizable output in Excel and CSV formats.
Table of Contents

Features
Installation
Usage
Branching Strategy
Screenshots
Output Structure
Project Structure
Requirements
Notes
License
Contributing

Features

Real-time Detection: Perform object detection on images, videos, or webcam feeds using YOLOv8.
Batch Processing: Process multiple images with results saved to images/, txt/, and user-selected Excel and/or CSV files.
System Monitoring: Display real-time CPU, GPU, and memory usage.
Customizable Confidence: Adjust detection confidence via synchronized sliders and spinbox.
Result Saving: Save individual detection results as images and text files in images/ and txt/ subfolders.
Progress Tracking: Visualize batch processing progress with a progress bar.
Logging: Log all operations (model loading, processing, errors) to logs.log for debugging.
Custom UI:
waiting.png: Default image displayed at startup and after resetting the image display.
loading.png: Shown during batch processing for better user experience.
icon.png: Custom application icon.


Batch Output Options: Select Excel, CSV, or both via checkboxes in the Batch Processing tab (at least one must be selected, or an error is shown).
Modular Code: Organized codebase with English documentation for maintainability.

Installation

Install Python 3.8+.
Clone the repository:git clone https://github.com/OrangeP1llow/object-classifier-project.git
cd object-classifier-project


Switch to the develop branch for the full codebase:git checkout develop


Install dependencies:pip install -r requirements.txt


Download the YOLOv8 model (yolov8x.pt) from Google Drive and place it in the project root.
Ensure the assets/ folder contains waiting.png, loading.png, and icon.png.

Usage
Run the application:
python main.py

Interface

Main Functions Tab:
Upload an image, video, or start the webcam.
Adjust the confidence threshold and initiate detection.
Save results to images/ and txt/ subfolders.


Batch Processing Tab:
Select input and output folders.
Choose output formats (Excel, CSV, or both) via checkboxes.
Process images and save results to images/, txt/, and selected Excel/CSV files.
Displays loading.png during processing and waiting.png by default or after completion.
Monitor progress with a progress bar.


About Tab:
View application information.



Logs
All operations are logged to logs.log in the project root, including:

Model loading
Image/video processing
Errors and warnings
Checkbox selections
Image display events

View logs:
cat logs.log  # On Linux/Mac
type logs.log  # On Windows

Branching Strategy

main: Contains only README.md and .gitignore for a clean project overview.
develop: Full codebase, including source files, assets, documentation, and README.md.

Screenshots

Batch Processing Tab (with loading.png and progress bar):
Default View (with waiting.png):

Output Structure

Single Detection:
Images: save_folder/images/detection_*.jpg
Text files: save_folder/txt/detection_*.txt


Batch Processing:
Images: output_folder/images/detected_*.jpg
Text files: output_folder/txt/detected_*.txt
Excel (if selected): output_folder/batch_results_*.xlsx
Columns: Filename, Objects Found, Processing Time (s), Classes Detected


CSV (if selected): output_folder/batch_results_*.csv
Same columns as Excel





Project Structure
Available on the develop branch:
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
Dependencies are listed in requirements.txt. Key libraries:

PyQt5
opencv-python
numpy
pandas
ultralytics
psutil
pynvml
torch

Install them:
pip install -r requirements.txt

Notes

The yolov8x.pt model (~100 MB) is not included due to size. Download it from the provided Google Drive link.
Ensure assets/ contains waiting.png, loading.png, and icon.png to avoid warnings in logs.log (e.g., WARNING - Default image 'waiting.png' not found).
Logs in logs.log provide detailed debugging information.

License
This project is developed for educational purposes and is not licensed for commercial use.
Contributing
Contributions are welcome! Please:

Create a feature branch from develop.
Submit a Pull Request for review.


Developed by OrangeP1llow
