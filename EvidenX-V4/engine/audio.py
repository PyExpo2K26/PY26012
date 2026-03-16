"""
engine/audio.py
---------------
Audio forensics analysis for EvidenX.

Visualisations returned (all base64-encoded PNG):
  1. stft_spec   — Traditional STFT Power Spectrogram (linear frequency scale)
  2. mel_spec    — Mel-Spectrogram (log frequency, perceptually scaled)
  3. waveform    — Time-domain waveform
  4. mfcc        — MFCC heatmap (13 coefficients)
  5. chroma      — Chromagram (pitch-class energy over time)

Score:
  LSTM-based fake probability (BiLSTM over STFT frames) blended with
  a spectral heuristic for calibration when no weights are loaded.
"""

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import io
import base64

from engine.audio_lstm import predict_lstm_from_stft

# ── Matplotlib style for all plots ────────────────────────────────────────────
_PLOT_STYLE = {
    "figure.facecolor": "#020204",
    "axes.facecolor":   "#0a0f14",
    "axes.edgecolor":   "#1a2f3a",
    "axes.labelcolor":  "#8899a6",
    "xtick.color":      "#8899a6",
    "ytick.color":      "#8899a6",
    "text.color":       "#e0faff",
    "grid.color":       "#1a2f3a",
    "grid.linestyle":   "--",
}


def _fig_to_b64(fig) -> str:
    """Serialise a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _style_axes(ax, title: str, colorbar=None):
    """Apply cybernetic theme to a matplotlib axes object."""
    ax.set_title(title, color="#00f3ff", fontsize=11, pad=8, fontweight="bold")
    ax.tick_params(colors="#8899a6", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2f3a")
    if colorbar:
        colorbar.ax.yaxis.set_tick_params(color="#8899a6", labelsize=7)
        plt.setp(colorbar.ax.yaxis.get_ticklabels(), color="#8899a6")


# ── 1. STFT Power Spectrogram ─────────────────────────────────────────────────

def _plot_stft(y: np.ndarray, sr: int) -> str:
    """Traditional short-time Fourier transform magnitude spectrogram."""
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        n_fft = 1024
        hop   = 256
        D     = librosa.amplitude_to_db(
                    np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)),
                    ref=np.max)
        img  = librosa.display.specshow(
                    D, sr=sr, hop_length=hop, x_axis="time",
                    y_axis="linear", ax=ax, cmap="inferno")
        cb   = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        _style_axes(ax, "STFT Power Spectrogram", cb)
        ax.set_xlabel("Time (s)", color="#8899a6")
        ax.set_ylabel("Frequency (Hz)", color="#8899a6")
        fig.tight_layout()
        return _fig_to_b64(fig)


# ── 2. Mel Spectrogram ────────────────────────────────────────────────────────

def _plot_mel(y: np.ndarray, sr: int) -> str:
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        S     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB  = librosa.power_to_db(S, ref=np.max)
        img   = librosa.display.specshow(S_dB, sr=sr, x_axis="time",
                    y_axis="mel", ax=ax, cmap="magma")
        cb    = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        _style_axes(ax, "Mel-Spectrogram", cb)
        ax.set_xlabel("Time (s)", color="#8899a6")
        ax.set_ylabel("Frequency (mel)", color="#8899a6")
        fig.tight_layout()
