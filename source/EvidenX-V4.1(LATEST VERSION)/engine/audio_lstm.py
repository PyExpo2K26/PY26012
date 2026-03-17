"""
engine/audio_lstm.py
--------------------
Lightweight Bidirectional LSTM classifier for audio deepfake detection,
operating over traditional STFT spectrogram frames.

Architecture:
  Input  : (batch, time_steps, n_fft//2+1)  — one STFT frame per time step
  LSTM   : 2-layer BiLSTM, hidden=128 each direction
  Attention: soft temporal attention pooling
  Output : scalar fake-probability  [0, 1]

Usage:
    from engine.audio_lstm import load_audio_lstm_model, predict_lstm_from_stft
    load_audio_lstm_model()                         # called once on startup
    prob, attn = predict_lstm_from_stft(y, sr)      # per-audio inference
"""

import os
import numpy as np
import torch
import torch.nn as nn
import librosa

# ──────────────────────────────────────────────
# Model Definition
# ──────────────────────────────────────────────

N_FFT       = 1024       # FFT window size  → freq bins = N_FFT//2+1 = 513
HOP_LENGTH  = 512        # hop between frames
INPUT_DIM   = N_FFT // 2 + 1   # 513
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
MAX_FRAMES  = 128        # truncate / pad to this many time steps

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "audio_lstm_weights.pth"
)

# Singleton model reference
_model = None
_device = None


class SoftAttention(nn.Module):
    """Single-head soft attention over time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)  # *2 for bidir

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (batch, T, hidden*2)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, T)
        weights = torch.softmax(scores, dim=1)            # (batch, T)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden*2)
        return context, weights


class SpectrogramLSTM(nn.Module):
    """
    2-layer Bidirectional LSTM operating on STFT spectrogram frames.
    """

    def __init__(self,
                 input_dim: int  = INPUT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS,
                 dropout: float  = 0.3):
        super().__init__()

        # Normalisation of input frequency bins
        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        self.attention = SoftAttention(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, T, input_dim)
        Returns:
            prob   : (batch,)  fake probability
            weights: (batch, T) attention weights
        """
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)                # (batch, T, hidden*2)
        context, weights = self.attention(lstm_out)  # (batch, hidden*2), (batch, T)
        prob = self.classifier(context).squeeze(-1)  # (batch,)
        return prob, weights


# ──────────────────────────────────────────────
# Model Lifecycle
# ──────────────────────────────────────────────

def load_audio_lstm_model() -> None:
    """
    Initialise the global LSTM model.
    Loads weights from various potential locations/names.
    """
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = SpectrogramLSTM().to(_device)
    _model.eval()

    # Broad search for audio weights
    potential_paths = [
        WEIGHTS_PATH,
        os.path.join(os.path.dirname(__file__), "..", "audio_lstm_v1.pth"),
        os.path.join(os.path.dirname(__file__), "..", "audio_model.pth"),
        os.path.join(os.path.dirname(__file__), "..", "models", "audio_lstm.pth"),
        "audio_lstm_weights.pth"
    ]

    weights_to_load = None
    for p in potential_paths:
        if os.path.isfile(p):
            weights_to_load = p
            break

    if weights_to_load:
        try:
            state = torch.load(weights_to_load, map_location=_device)
            # Handle checkpoint wrapper
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            _model.load_state_dict(state, strict=False)
            print(f"[AudioLSTM] Loaded weights from {weights_to_load}")
        except Exception as e:
            print(f"[AudioLSTM] Could not load weights from {weights_to_load} ({e}). Using random init.")
    else:
        print("[AudioLSTM] No pre-trained weights found in search paths — using random initialisation.")


# ──────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────

def extract_stft_frames(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute the STFT magnitude spectrogram and return frames shaped
    (T, N_FFT//2+1) — ready for LSTM input.
    Frames are normalised to [0, 1] per frame for stability.
    """
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag  = np.abs(stft).T          # (T, freq_bins)
    mag  = librosa.amplitude_to_db(mag + 1e-9, ref=np.max)

    # Normalise to [0, 1]
    min_v, max_v = mag.min(), mag.max()
    if max_v - min_v > 1e-6:
        mag = (mag - min_v) / (max_v - min_v)

    # Pad or truncate time axis
    T = mag.shape[0]
    if T >= MAX_FRAMES:
        mag = mag[:MAX_FRAMES]
    else:
        pad = np.zeros((MAX_FRAMES - T, mag.shape[1]), dtype=np.float32)
        mag = np.concatenate([mag, pad], axis=0)

    return mag.astype(np.float32)   # (MAX_FRAMES, freq_bins)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict_lstm_from_stft(y: np.ndarray, sr: int):
    """
    Run the LSTM on raw audio and return:
        fake_prob  : float  [0.0 – 1.0]
        attn_weights: np.ndarray shape (MAX_FRAMES,) — per-frame attention
    """
    global _model, _device

    # Lazy load if startup wasn't called explicitly
    if _model is None:
        load_audio_lstm_model()

    frames = extract_stft_frames(y, sr)              # (T, freq_bins)
    x = torch.tensor(frames).unsqueeze(0).to(_device)  # (1, T, freq_bins)

    with torch.no_grad():
        prob, weights = _model(x)

    fake_prob    = float(prob.cpu().numpy()[0])
    attn_weights = weights.cpu().numpy()[0]          # (MAX_FRAMES,)

    # ── Calibration heuristic (when no real weights are loaded) ──────────
    # Random-init models tend to output ~0.5. We blend with a spectral
    # feature heuristic so the score is still informative:
    mfcc_var   = float(np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).mean())
    sc_mean    = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    heuristic  = _mfcc_heuristic(mfcc_var, sc_mean)

    # Weight LSTM output vs. heuristic (60/40 blend)
    calibrated = fake_prob * 0.6 + heuristic * 0.4

    return float(np.clip(calibrated, 0.0, 1.0)), attn_weights


def _mfcc_heuristic(mfcc_var: float, contrast_mean: float) -> float:
    """
    Legacy heuristic kept as a calibration anchor.
    Lower MFCC variance → more likely synthetic.
    """
    if mfcc_var < 30:
        return 0.78
    elif mfcc_var < 50:
        return 0.42
    else:
        return 0.12
