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

