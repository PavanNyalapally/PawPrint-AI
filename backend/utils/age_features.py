# backend/utils/age_features.py

import cv2
import numpy as np


def extract_footprint_area(image_path: str, target_size=(224, 224)) -> float:
    """
    Extract approximate footprint area using contour detection.

    - Image is resized to a fixed size for consistency
    - Returns area in pixels (resized space)
    - Safe for Flask inference (never crashes)
    """

    image = cv2.imread(image_path)

    if image is None:
        return 0.0

    # Resize for consistency with model input
    image = cv2.resize(image, target_size)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary threshold (more stable than adaptive for outdoor images)
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological operations to clean noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    # Select the largest reasonable contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    return float(area)
