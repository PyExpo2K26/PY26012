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
        return _fig_to_b64(fig)


# ── 3. Waveform ───────────────────────────────────────────────────────────────

def _plot_waveform(y: np.ndarray, sr: int) -> str:
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.8, color="#00f3ff")
        _style_axes(ax, "Waveform (Time Domain)")
        ax.set_xlabel("Time (s)", color="#8899a6")
        ax.set_ylabel("Amplitude", color="#8899a6")
        ax.fill_between(
            np.linspace(0, len(y) / sr, len(y)), y, 0,
            where=(np.abs(y) > 0.3 * np.max(np.abs(y))),
            color="#ff003c", alpha=0.25, label="High-energy regions"
        )
        fig.tight_layout()
        return _fig_to_b64(fig)


# ── 4. MFCC Heatmap ───────────────────────────────────────────────────────────

def _plot_mfcc(y: np.ndarray, sr: int) -> str:
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img   = librosa.display.specshow(mfccs, x_axis="time", ax=ax, cmap="coolwarm")
        cb    = fig.colorbar(img, ax=ax)
        _style_axes(ax, "MFCC Feature Map (13 Coefficients)", cb)
        ax.set_xlabel("Time (s)", color="#8899a6")
        ax.set_ylabel("MFCC Coefficient", color="#8899a6")
        fig.tight_layout()
        return _fig_to_b64(fig)


# ── 5. Chromagram ─────────────────────────────────────────────────────────────

def _plot_chroma(y: np.ndarray, sr: int) -> str:
    """Chroma-STFT: reveals pitch-class energy — useful for detecting TTS artifacts."""
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        chroma  = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        img     = librosa.display.specshow(chroma, y_axis="chroma", x_axis="time",
                      ax=ax, cmap="YlOrRd")
        cb      = fig.colorbar(img, ax=ax)
        _style_axes(ax, "Chromagram (Pitch Class Energy)", cb)
        ax.set_xlabel("Time (s)", color="#8899a6")
        ax.set_ylabel("Pitch Class", color="#8899a6")
        fig.tight_layout()
        return _fig_to_b64(fig)


# ── 6. Attention Heatmap (LSTM) ───────────────────────────────────────────────

def _plot_attention(attn_weights: np.ndarray, sr: int, duration: float) -> str:
    """
    Visualise per-frame LSTM attention weights as a 1-D heatmap
    overlaid on a time axis — highlights which audio segments the
    model flagged as most suspicious.
    """
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 1.8))
        # attn_weights shape: (MAX_FRAMES,)
        w = attn_weights / (attn_weights.max() + 1e-9)  # normalise
        t_axis = np.linspace(0, duration, len(w))

        ax.fill_between(t_axis, w, color="#00f3ff", alpha=0.6)
        ax.plot(t_axis, w, color="#00f3ff", linewidth=1)

        # Shade top-10% attention frames in danger red
        threshold = np.percentile(w, 90)
        ax.fill_between(t_axis, w, where=(w >= threshold),
                         color="#ff003c", alpha=0.7, label="High-attention")

        _style_axes(ax, "LSTM Temporal Attention (Suspicious Regions)")
        ax.set_xlabel("Time (s)", color="#8899a6")
        ax.set_ylabel("Attention", color="#8899a6")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, labelcolor="#8899a6",
                  facecolor="#0a0f14", edgecolor="#1a2f3a")
        fig.tight_layout()
        return _fig_to_b64(fig)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def analyze_audio(file_path: str):
    """
    Full audio analysis pipeline.

    Returns:
        score           : float [0,1]  — overall fake probability
        stft_spec_b64   : str  — base64 PNG of STFT spectrogram
        mel_spec_b64    : str  — base64 PNG of Mel-spectrogram
        waveform_b64    : str  — base64 PNG of waveform
        mfcc_b64        : str  — base64 PNG of MFCC heatmap
        chroma_b64      : str  — base64 PNG of chromagram
        attn_b64        : str  — base64 PNG of LSTM attention
    """
    try:
        # Load audio (cap at 15 s for speed)
        y, sr = librosa.load(file_path, duration=15, sr=None, mono=True)

        # ── Generate all visualisations in parallel-friendly order ──────
        stft_b64    = _plot_stft(y, sr)
        mel_b64     = _plot_mel(y, sr)
        waveform_b64 = _plot_waveform(y, sr)
        mfcc_b64    = _plot_mfcc(y, sr)
        chroma_b64  = _plot_chroma(y, sr)

        # ── LSTM Inference ──────────────────────────────────────────────
        score, attn_weights = predict_lstm_from_stft(y, sr)
        attn_b64 = _plot_attention(attn_weights, sr, len(y) / sr)

        return score, stft_b64, mel_b64, waveform_b64, mfcc_b64, chroma_b64, attn_b64

    except Exception as e:
        print(f"[AudioAnalysis] Error: {e}")
        # Return safe neutral defaults on failure
        return 0.0, "", "", "", "", "", ""
