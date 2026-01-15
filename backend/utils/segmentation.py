# backend/utils/segmentation.py

import cv2
import numpy as np


def segment_footprint(image_path: str, output_size=(224, 224)) -> np.ndarray:
    """
    Segments the footprint region from background.

    - Robust to noisy outdoor backgrounds
    - Flask-safe (never raises)
    - Returns RGB image resized to output_size
    """

    image = cv2.imread(image_path)
    if image is None:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        resized = cv2.resize(image, output_size)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    largest = max(contours, key=cv2.contourArea)

    img_area = image.shape[0] * image.shape[1]
    if cv2.contourArea(largest) < 100 or cv2.contourArea(largest) > 0.9 * img_area:
        resized = cv2.resize(image, output_size)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    x, y, w, h = cv2.boundingRect(largest)
    cropped = image[y:y + h, x:x + w]
    cropped = cv2.resize(cropped, output_size)

    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
