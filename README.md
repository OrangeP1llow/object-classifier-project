### **YOLO Object Classifier Project**

### Project Overview
This repository contains the implementation for a **YOLO Object Detection** application. The goal of the project is to create a fully functional GUI-based application that performs real-time object detection using the **YOLOv8** model. The application supports processing images, videos, webcam feeds, and batch operations, with customizable output formats and a user-friendly interface.

### The development of this project consists of several key steps:
1. Designing and developing the object detection logic using YOLOv8.
2. Utilizing PyQt5 to create a graphical user interface (GUI).
3. Handling input/output interactions through file uploads, webcam feeds, and result saving.
4. Ensuring seamless functionality across the following features:
   - **Single Detection: Process images, videos, or webcam feeds with adjustable confidence thresholds.**
   - **Batch Processing: Process multiple images with Excel/CSV output.**
   - **System Monitoring: Display real-time CPU, GPU, and memory usage.**
   - **Logging: Record all operations to logs.log for debugging.**

### Technologies and Software
#### Technologies:
- **Programming Language**: Python for application logic and GUI.
- **Frameworks and Libraries**:
    - **PyQt5: For the graphical user interface.**
    - **Ultralytics YOLO: For object detection.**
    - **OpenCV: For image and video processing.**
    - **Pandas: For Excel/CSV output.**
    - **Psutil and Pynvml: For system monitoring.**
- **Development Environment**: Any Python-compatible IDE (e.g., VS Code, PyCharm).
- **Version Control**: Git for source code management and collaboration.
  
#### Software Requirements:
- **Python**: Version 3.8 or higher.
- **YOLOv8 Model**: **yolov8x.pt** (downloaded separately).
- **Dependencies**: Listed in **requirements.txt**.

#### Installation:
1. Install **Python 3.8+** from python.org.
2. Clone the repository
3. Switch to the **develop** branch for the full codebase
4. Install dependencies
5. Download the **YOLOv8 model** **(yolov8x.pt)** from Google Drive and place it in the project root.
6. Ensure the **assets/** folder contains **waiting.png**, **loading.png**, and **icon.png**.

#### Usage:
- **Run the application**
- **Interface**:
   - **Main Functions Tab:**
      - Upload an image, video, or start the webcam.
      - Adjust the confidence threshold via synchronized sliders and spinbox.
      - Save detection results to images/ and txt/ subfolders.
   - **Batch Processing Tab:**
      - Select input and output folders.
      - Choose output formats (**Excel**, **CSV**, or both) via checkboxes (at least one must be selected, or an error is shown).
      - Process multiple images and save results to **images/**, **txt/**, and selected **Excel/CSV** files.
      - Displays **loading.png** during processing and **waiting.png** by default or after completion/cancellation.
      - Monitor progress with a progress bar.
    - **About Tab:**
      - View information about the application.

#### Branching Strategy:
- **main**: Contains only **README.md** and **.gitignore** for a clean project overview.
- **develop**: Full codebase, including source files, assets, documentation, and README.md.

#### Notes:
- The **yolov8x.pt** model (~100 MB) is not included in the repository due to its size.
- Ensure the **assets/** folder contains **waiting.png**, **loading.png**, and **icon.png** to avoid warnings in logs.log (e.g., WARNING - Default image 'waiting.png' not found).
- Logs in **logs.log** provide detailed information for debugging.

#### License:
- This project is developed for educational purposes and is not licensed for commercial use.

Developed by OrangeP1llow with ❤
---