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

