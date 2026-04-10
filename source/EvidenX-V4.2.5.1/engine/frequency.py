"""
engine/frequency.py
====================
Frequency-domain forensic analysis using:
  1. FFT (Fast Fourier Transform) — detects GAN / diffusion spectral fingerprints
     that appear as periodic grid noise patterns in the power spectrum.
  2. DCT (Discrete Cosine Transform) block artifact analysis — detects
     double-JPEG-compression signatures (image splicing).

Returns
-------
score       : float  — combined forgery score in [0, 1]
fft_image   : str    — base64 PNG of the FFT power spectrum (log-scaled, colormapped)
dct_image   : str    — base64 PNG of the DCT block artifact map
"""
import cv2
import numpy as np
import base64
import io


# ── helpers ────────────────────────────────────────────────────────────────────
def _to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode()


# ── 1. FFT Analysis ─────────────────────────────────────────────────────────────
def _fft_analysis(gray: np.ndarray):
    """
    Returns (score, fft_img_bgr)
    - score: fraction of high-frequency energy in the power spectrum.
    GAN images have unnatural high-freq spikes → higher score.
    """
    h, w = gray.shape

    # Zero-pad to next power of two for performance
    optimal_h = cv2.getOptimalDFTSize(h)
    optimal_w = cv2.getOptimalDFTSize(w)
    padded = np.zeros((optimal_h, optimal_w), dtype=np.float32)
    padded[:h, :w] = gray.astype(np.float32)

    # Compute FFT
    dft = np.fft.fft2(padded)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)

    # Log-scale for display
    log_mag = np.log1p(magnitude)
    log_mag_norm = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Colormap
    fft_colored = cv2.applyColorMap(log_mag_norm, cv2.COLORMAP_MAGMA)

    # ── Score: ratio of high-frequency energy ────────────────────────────
    cy, cx = optimal_h // 2, optimal_w // 2
    # Low-frequency radius: inner 10% of the spectrum
    radius = int(min(cy, cx) * 0.10)

    # Create radial mask
    Y, X = np.ogrid[:optimal_h, :optimal_w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    low_freq_mask = dist <= radius

    total_energy  = np.sum(magnitude)
    low_f_energy  = np.sum(magnitude[low_freq_mask])
    high_f_energy = total_energy - low_f_energy

    # Natural photos concentrate energy in low-freq; GANs spray more energy
    high_f_ratio = high_f_energy / (total_energy + 1e-9)

    # Calibrate: natural photos ~0.92 high-freq (just normalisation), GANs deviate
    # We look for unnatural *spikes* instead: count pixels above 3-sigma threshold
    fft_flat = log_mag.flatten()
    threshold = np.mean(fft_flat) + 3 * np.std(fft_flat)
    spike_fraction = np.sum((log_mag > threshold) & ~low_freq_mask) / (optimal_h * optimal_w)

    # Final score: heavier weight on spike fraction
    fft_score = float(np.clip(spike_fraction * 6.0, 0.0, 1.0))

    return fft_score, fft_colored


# ── 2. DCT Block Artifact Analysis ─────────────────────────────────────────────
def _dct_analysis(gray: np.ndarray):
    """
    Returns (score, dct_img_bgr)
    Double-JPEG compression leaves a characteristic quantization grid at block
    boundaries (8×8 blocks). We compute the variance at block edges vs. interior.
    """
    h, w = gray.shape
    block = 8
    img_f = gray.astype(np.float32)

    # Compute horizontal and vertical edge maps
    diff_h = np.abs(np.diff(img_f, axis=1))   # (H, W-1)
    diff_v = np.abs(np.diff(img_f, axis=0))   # (H-1, W)

    # Mask for block-boundary columns (every 8th column)
    boundary_cols = np.zeros(w - 1, dtype=bool)
    boundary_cols[(np.arange(w - 1) + 1) % block == 0] = True
    boundary_rows = np.zeros(h - 1, dtype=bool)
    boundary_rows[(np.arange(h - 1) + 1) % block == 0] = True

    edge_var_h   = np.mean(diff_h[:, boundary_cols])
    inter_var_h  = np.mean(diff_h[:, ~boundary_cols])
    edge_var_v   = np.mean(diff_v[boundary_rows, :])
    inter_var_v  = np.mean(diff_v[~boundary_rows, :])

    # Periodicity ratio: >1 means block boundaries stand out (double compression)
    ratio_h = edge_var_h / (inter_var_h + 1e-9)
    ratio_v = edge_var_v / (inter_var_v + 1e-9)
    periodicity = (ratio_h + ratio_v) / 2.0

    # Score: 1.0 → ratio of 2.0 or above (heavy block artifact)
    dct_score = float(np.clip((periodicity - 1.0) / 1.5, 0.0, 1.0))

    # Visualization: build a block-artifact heat map
    # Mark block boundaries with a heat overlay
    artifact_map = np.zeros_like(gray, dtype=np.float32)
    for col in range(0, w, block):
        artifact_map[:, col] = diff_h.mean(axis=0)[min(col, w - 2)]
    for row in range(0, h, block):
        artifact_map[row, :] = diff_v.mean(axis=1)[min(row, h - 2)]

    norm_map = cv2.normalize(artifact_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dct_colored = cv2.applyColorMap(norm_map, cv2.COLORMAP_INFERNO)

    return dct_score, dct_colored


# ── Public API ──────────────────────────────────────────────────────────────────
def analyze_frequency(image_path: str):
    """
    Runs FFT + DCT analysis on an image file.

    Returns
    -------
    score      : float — combined frequency-domain forgery score (0–1)
    fft_b64    : str   — base64 PNG, FFT power spectrum
    dct_b64    : str   — base64 PNG, DCT block-artifact map
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot open image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fft_score, fft_img = _fft_analysis(gray)
        dct_score, dct_img = _dct_analysis(gray)

        # Weighted combined score
        combined = fft_score * 0.55 + dct_score * 0.45

        return float(np.clip(combined, 0.0, 1.0)), _to_b64(fft_img), _to_b64(dct_img)

    except Exception as e:
        print(f"Frequency Analysis Error: {e}")
        return 0.0, "", ""
