import cv2
import numpy as np
from ultralytics import YOLO


# ==========================================
# Model
# ==========================================

# Path to the trained YOLO model
MODEL_PATH = "best_yolo26l.pt"

print("==========================================")
print("Loading YOLO model...")
print("CPU mode - this may take some time.")
print("Please wait...")
print("==========================================")

# Load the trained YOLO model
model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")
print("==========================================")


# ==========================================
# Detect Clothing
# ==========================================

def detect_clothing(image, conf_threshold=0.35):
    """
    Detect clothing items in the input image.

    Parameters:
        image: Input image as a NumPy array.
        conf_threshold: Minimum confidence score required
                        to keep a detection.

    Returns:
        A list containing the detected clothing items,
        their confidence scores, and bounding boxes.
    """

    # Run YOLO object detection on the input image
    results = model.predict(
        source=image,
        conf=conf_threshold,
        device="cpu",
        verbose=False
    )

    # Get the first prediction result by cpu
    result = results[0]

    # Create an empty list to store detected items
    detected_items = []

    # Return an empty list if no bounding boxes were detected
    if result.boxes is None:
        return detected_items

    # Get the class names from the YOLO model
    names = result.names

    # Loop through all detected objects
    for i, box in enumerate(result.boxes):

        # Get the confidence score of the detection
        confidence = float(box.conf[0])

        # Get the class ID of the detected object
        class_id = int(box.cls[0])

        # Convert the class ID into the corresponding class name
        class_name = names[class_id]

        # Extract the bounding box coordinates
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Store the detection information in a dictionary
        detected_items.append({
            "id": i,
            "item": class_name,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })

    # Return all detected clothing items
    return detected_items


# ==========================================
# Crop Detected Items
# ==========================================

def crop_detected_items(image, detected_items):
    """
    Crop each detected clothing item from the original image.

    Parameters:
        image: Original input image.
        detected_items: List of detected objects and their bounding boxes.

    Returns:
        A list containing the cropped images and detection information.
    """

    # Create an empty list to store cropped clothing items
    cropped_items = []

    # Get the image dimensions
    height, width = image.shape[:2]

    # Loop through all detected clothing items
    for item in detected_items:

        # Get the bounding box coordinates
        x1, y1, x2, y2 = item["bbox"]

        # Make sure the coordinates are inside the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        # Crop the detected object from the image
        crop = image[y1:y2, x1:x2]

        # Skip the crop if it is empty or invalid
        if crop.size == 0:
            continue

        # Store the cropped image along with its detection information
        cropped_items.append({
            **item,
            "crop": crop
        })

    # Return all valid cropped clothing items
    return cropped_items


# ==========================================
# Draw Detection Boxes
# ==========================================

def draw_detections(image, detected_items):
    """
    Draw bounding boxes and labels around detected clothing items.

    Parameters:
        image: Original input image.
        detected_items: List of detected objects.

    Returns:
        The image with bounding boxes and labels.
    """

    # Create a copy so the original image is not modified
    output = image.copy()

    # Loop through all detected items
    for item in detected_items:

        # Get the bounding box coordinates
        x1, y1, x2, y2 = item["bbox"]

        # Create a label containing the item name
        # and its confidence percentage
        label = (
            f'{item["item"]} '
            f'{item["confidence"] * 100:.1f}%'
        )

        # Draw the bounding box around the detected item
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw the class name and confidence score above the box
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Return the image with the drawn detections
    return output