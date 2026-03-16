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
