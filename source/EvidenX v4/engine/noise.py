"""
engine/noise.py
================
PRNU-inspired noise residual analysis for forgery detection.

Photo Response Non-Uniformity (PRNU): every camera sensor has a unique
noise pattern imprinted on every image it captures. When an image is
digitally composited or spliced, the noise pattern becomes inconsistent
across regions — a reliable forgery signal.

We approximate PRNU analysis using:
  1. Median-filter residuals: noise = original - median_filtered
  2. Multi-scale noise variance mapping
  3. Regional variance inconsistency detection

Returns
-------
score      : float — forgery score in [0, 1] based on noise inconsistency
noise_b64  : str   — base64 PNG, multi-scale noise map (false-colour)
"""
import cv2
import numpy as np
import base64


def _to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode()


def _noise_residual(channel: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Extract high-frequency noise residual via median filtering."""
    blurred = cv2.medianBlur(channel, ksize)
    residual = channel.astype(np.float32) - blurred.astype(np.float32)
    return residual


def _block_variance_map(residual: np.ndarray, block: int = 32) -> np.ndarray:
    """
    Compute per-block variance of noise residual.
    Returns a variance map (same spatial shape but block-resolution).
    """
    h, w = residual.shape
    bh = h // block
    bw = w // block
    var_map = np.zeros((bh, bw), dtype=np.float32)

    for r in range(bh):
        for c in range(bw):
            patch = residual[r * block:(r + 1) * block, c * block:(c + 1) * block]
            var_map[r, c] = np.var(patch)

    return var_map


def analyze_noise(image_path: str):
    """
    PRNU noise residual analysis.

    Parameters
    ----------
    image_path : str  — path to the image file

    Returns
    -------
    score     : float — inconsistency score (0 = natural, 1 = highly suspect)
    noise_b64 : str   — base64 PNG of the false-colour noise inconsistency map
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot open image: {image_path}")

        h, w = img.shape[:2]
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)))

        h, w = img.shape[:2]
        channels = cv2.split(img)

        combined_residual = np.zeros((h, w), dtype=np.float32)
        for ch in channels:
            for ksize in (3, 5, 7):
                combined_residual += np.abs(_noise_residual(ch, ksize))
        combined_residual /= (3 * 3)

        block = max(16, min(h, w) // 16)
        var_map = _block_variance_map(combined_residual, block=block)

        mean_var = np.mean(var_map) + 1e-9
        std_var  = np.std(var_map)
        coeff_of_variation = std_var / mean_var 

        score = float(np.clip((coeff_of_variation - 0.5) / 1.5, 0.0, 1.0))

        var_vis = cv2.resize(var_map, (w, h), interpolation=cv2.INTER_NEAREST)
        var_norm = cv2.normalize(var_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        noise_colored = cv2.applyColorMap(var_norm, cv2.COLORMAP_HOT)
        blended = cv2.addWeighted(img, 0.45, noise_colored, 0.55, 0)

        return score, _to_b64(blended)

    except Exception as e:
        print(f"Noise Analysis Error: {e}")
        return 0.0, ""


