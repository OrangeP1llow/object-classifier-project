import cv2
from datetime import datetime

def save_detection_results(image, filename):
    """
    Save an image with detection results.

    Args:
        image (np.ndarray): Image to save.
        filename (str): Path to save the image.

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        if image is None:
            return False
        return cv2.imwrite(filename, image)
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        return False

def save_txt_file(txt_path, filename, processing_time, detection_results):
    """
    Save detection details to a text file.

    Args:
        txt_path (str): Path to save the text file.
        filename (str): Name of the processed image file.
        processing_time (float): Time taken for detection in seconds.
        detection_results (list): List of detection dictionaries.

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        with open(txt_path, 'w') as f:
            f.write(f"Image: {filename}\n")
            f.write(f"Processing time: {processing_time:.3f} seconds\n")
            f.write(f"Total objects found: {len(detection_results)}\n\n")
            for i, detection in enumerate(detection_results):
                class_name = detection["class_name"]
                confidence = detection["confidence_percent"]
                x1, y1, x2, y2 = detection["box"]
                f.write(f"{i + 1}. {class_name} ({confidence:.1f}%)\n")
                f.write(f"   Position: ({x1}, {y1}) - ({x2}, {y2})\n")
        return True
    except Exception as e:
        print(f"Error saving text file {txt_path}: {e}")
        return False

def generate_unique_filename(prefix="detection", extension=".jpg"):
    """
    Generate a unique filename based on the current timestamp.

    Args:
        prefix (str): Prefix for the filename (default: "detection").
        extension (str): File extension (default: ".jpg").

    Returns:
        str: Unique filename, e.g., "detection_20250426_123456_123456.jpg".
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}{extension}"