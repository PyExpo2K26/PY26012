import os
from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
import base64
import io

def perform_ela(image_path, quality=90):
    """
    Performs Error Level Analysis (ELA) on the image using OpenCV for advanced heatmap visualization.
    Returns:
        ela_image_base64: Base64 encoded ELA heatmap (JET Colormap).
        score: A heuristic score representing the amount of noise (potential tampering).
    """
    try:
        original_cv = cv2.imread(image_path)
        if original_cv is None:
             raise Exception("Could not read image with OpenCV")

        temp_filename = "temp_ela.jpg"
        cv2.imwrite(temp_filename, original_cv, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        compressed_cv = cv2.imread(temp_filename)
        
        diff = cv2.absdiff(original_cv, compressed_cv)
        
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        max_diff = np.max(gray_diff)
        if max_diff == 0:
            scale = 1
        else:
            scale = 255.0 / max_diff

        ela_image = cv2.convertScaleAbs(gray_diff, alpha=scale)
        
        heatmap = cv2.applyColorMap(ela_image, cv2.COLORMAP_JET)
        
        score = np.mean(gray_diff) / 255.0
        
        os.remove(temp_filename)

        _, buffer = cv2.imencode('.png', heatmap)
        img_str = base64.b64encode(buffer).decode()
        
        return img_str, float(score)
    
    except Exception as e:
        print(f"ELA Error: {e}")
        return None, 0.0




             
