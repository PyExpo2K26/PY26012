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
        # Read original image
        original_cv = cv2.imread(image_path)
        if original_cv is None:
             raise Exception("Could not read image with OpenCV")
             
