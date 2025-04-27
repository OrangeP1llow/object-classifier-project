import sys
from PyQt5.QtWidgets import QApplication
from gui import YOLODetectorGUI
from detector import YOLODetector

def main():
    """
    Entry point for the YOLO Object Detection application.
    Initializes the Qt application, YOLO detector, and GUI.
    """
    try:
        app = QApplication(sys.argv)

        detector = YOLODetector()

        gui = YOLODetectorGUI(detector)

        gui.show()

        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()