import sys
import os
sys.path.append(r"c:\Users\KiTE\Desktop\EvidenX v4.5.2\Files")

modules = [
    "engine.ela",
    "engine.metadata",
    "engine.copymove",
    "engine.cnn",
    "engine.audio",
    "engine.audio_lstm",
    "engine.gan",
    "engine.diffusion",
    "engine.email",
    "engine.pcap",
    "engine.frequency",
    "engine.noise",
    "engine.hybrid_model"
]

print("Testing imports...")
success_count = 0
fail_count = 0

for m in modules:
    try:
        __import__(m)
        print(f"[OK] {m}")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] {m}: {e}")
        fail_count += 1

print(f"\nSummary: {success_count} success, {fail_count} failed.")
