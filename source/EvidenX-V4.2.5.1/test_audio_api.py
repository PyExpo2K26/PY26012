"""
test_audio_api.py
-----------------
Quick smoke-test for the /analyze_audio endpoint.
Generates a 2-second 440 Hz sine-wave WAV in memory and
POSTs it to the running EvidenX server.
"""
import io
import struct
import wave
import requests
import math
import sys

BASE_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Generate a synthetic WAV (no soundfile/scipy dependency)
# ---------------------------------------------------------------------------
def make_sine_wav(freq=440.0, duration=2.0, sr=16000) -> bytes:
    n_samples = int(sr * duration)
    samples   = [int(32767 * math.sin(2 * math.pi * freq * i / sr))
                 for i in range(n_samples)]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_analyze_audio():
    print("Generating synthetic 440 Hz sine-wave WAV (2 s, 16 kHz)...")
    wav_bytes = make_sine_wav()
    print(f"  WAV size: {len(wav_bytes)} bytes")

    print(f"\nPOSTing to {BASE_URL}/analyze_audio ...")
    try:
        resp = requests.post(
            f"{BASE_URL}/analyze_audio",
            files={"file": ("test_sine.wav", wav_bytes, "audio/wav")},
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        print("  ERROR: Could not connect. Is the server running? (python main.py)")
        sys.exit(1)

    print(f"  HTTP status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"  FAILED — response body: {resp.text[:500]}")
        sys.exit(1)

    data = resp.json()

    # ── Required keys ────────────────────────────────────────────────────────
    required_keys = [
        "fake_probability",
        "stft_image",
        "spectrum_image",    # Mel-spectrogram (kept old key for compat)
        "waveform_image",
        "mfcc_image",
        "chroma_image",
        "attn_image",
        "explanation",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        print(f"  FAILED — missing keys in response: {missing}")
        sys.exit(1)

    # ── Value sanity checks ──────────────────────────────────────────────────
    fp = data["fake_probability"]
    assert isinstance(fp, (int, float)) and 0 <= fp <= 100, \
        f"fake_probability out of range: {fp}"

    for key in ["stft_image", "spectrum_image", "waveform_image",
                "mfcc_image", "chroma_image", "attn_image"]:
        assert isinstance(data[key], str) and len(data[key]) > 100, \
            f"Key '{key}' appears to be empty or not base64"

    assert isinstance(data["explanation"], list) and len(data["explanation"]) >= 1, \
        "explanation list is empty"

    print("\n  ✓ All assertions passed!")
    print(f"  fake_probability : {fp}%")
    print(f"  explanation      : {data['explanation']}")
    print(f"  stft_image size  : {len(data['stft_image'])} chars (base64)")
    print(f"  chroma_image size: {len(data['chroma_image'])} chars (base64)")
    print(f"  attn_image size  : {len(data['attn_image'])} chars (base64)")
    print("\n  Audio LSTM pipeline: PASS ✓")


if __name__ == "__main__":
    test_analyze_audio()
