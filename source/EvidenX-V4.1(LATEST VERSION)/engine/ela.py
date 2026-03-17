import os
from PIL import Image, ImageChops, ImageEnhance
import cv2
import numpy as np
import base64
import io

def perform_ela(image_path, qualities=(70, 80, 90)):
    """
    Performs multi-quality Error Level Analysis (ELA).
    Re-compresses at 3 different quality levels and averages the
    difference maps, then normalises via the 99th-percentile for a
    more robust score that is less affected by high-frequency textures.

    Returns:
        ela_image_base64: Base64 encoded ELA heatmap (JET Colormap).
        score: Heuristic score (0-1) representing tampering probability.
    """
    try:
        original_cv = cv2.imread(image_path)
        if original_cv is None:
            raise Exception("Could not read image with OpenCV")

        combined_diff = np.zeros(original_cv.shape[:2], dtype=np.float64)

        for quality in qualities:
            temp_filename = f"temp_ela_{quality}.jpg"
            cv2.imwrite(temp_filename, original_cv,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            compressed_cv = cv2.imread(temp_filename)
            os.remove(temp_filename)

            diff = cv2.absdiff(original_cv, compressed_cv)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            combined_diff += gray_diff.astype(np.float64)

        # Average across quality levels
        combined_diff /= len(qualities)

        # ── Normalize via 99th percentile (robust to bright natural textures) ──
        p99 = np.percentile(combined_diff, 99)
        if p99 == 0:
            p99 = 1.0
        scaled = np.clip(combined_diff / p99 * 255.0, 0, 255).astype(np.uint8)

        # Colourmap
        heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

        # Score: mean of top-5% brightest pixels (more sensitive to hot spots)
        top5_threshold = np.percentile(combined_diff, 95)
        hot_pixels = combined_diff[combined_diff >= top5_threshold]
        score = float(np.clip(np.mean(hot_pixels) / 255.0, 0.0, 1.0))

        _, buffer = cv2.imencode('.png', heatmap)
        img_str = base64.b64encode(buffer).decode()

        return img_str, score

    except Exception as e:
        print(f"ELA Error: {e}")
        return None, 0.0


