# backend/utils/preprocess.py

import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


def preprocess_image(image_path: str, target_size=(224, 224)):
    """
    Load and preprocess an image for EfficientNetV2-S inference.

    IMPORTANT:
    - MUST match the training pipeline
    - Uses EfficientNetV2 preprocess_input
    - NO manual /255 scaling

    Args:
        image_path (str): Path to image file
        target_size (tuple): (width, height)

    Returns:
        np.ndarray | None:
            Preprocessed image tensor of shape (1, H, W, 3)
    """

    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image
