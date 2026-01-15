import cv2
import numpy as np
import base64

def generate_debug_overlay(image_path: str, target_size=(224, 224)) -> str:
    """
    Generate a base64 encoded image with debug contours overlay.
    """
    image = cv2.imread(image_path)
    if image is None:
        return ""

    # Resize to match model input / consistency
    image = cv2.resize(image, target_size)
    
    # Process exactly as age_features.py does
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    visual_img = image.copy()
    
    if contours:
        # Draw all contours in faint blue
        cv2.drawContours(visual_img, contours, -1, (255, 0, 0), 1)
        
        # Draw largest contour in bright green (the one used for area)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(visual_img, [largest_contour], -1, (0, 255, 0), 2)
        
        # Add text
        area = cv2.contourArea(largest_contour)
        cv2.putText(visual_img, f"Area: {int(area)}px", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Convert to base64
    _, buffer = cv2.imencode('.jpg', visual_img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{b64_str}"
