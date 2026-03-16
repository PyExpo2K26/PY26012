import os
import sys

print("--- Testing Imports ---")
try:
    import cv2
    print("OpenCV imported successfully.")
except ImportError as e:
    print(f"OpenCV Import Failed: {e}")

try:
    import torch
    print("PyTorch imported successfully.")
except ImportError as e:
    print(f"PyTorch Import Failed: {e}")

try:
    import librosa
    print("Librosa imported successfully.")
except ImportError as e:
    print(f"Librosa Import Failed: {e}")

print("\n--- Testing Engine Modules ---")
# Create a dummy image for testing
import numpy as np
import cv2

dummy_img_path = "debug_test_image.jpg"
# Create a simple noise image
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
cv2.imwrite(dummy_img_path, img)
print(f"Created {dummy_img_path}")

try:
    print("\n1. Testing ELA...")
    from engine.ela import perform_ela
    img_b64, score = perform_ela(dummy_img_path)
    print(f"ELA Score: {score}")
    print(f"ELA Image generated: {len(img_b64) > 0}")
except Exception as e:
    print(f"ELA Failed: {e}")

try:
    print("\n2. Testing Copy-Move...")
    from engine.copymove import detect_copymove
    count, score = detect_copymove(dummy_img_path)
    print(f"Copy-Move Count: {count}, Score: {score}")
except Exception as e:
    print(f"Copy-Move Failed: {e}")

try:
    print("\n3. Testing CNN (MesoNet)...")
    from engine.cnn import predict_cnn, load_model
    # Attempt to load model explicitly
    load_model()
    score, heatmap = predict_cnn(dummy_img_path)
    print(f"CNN Score: {score}")
    print(f"CNN Heatmap generated: {len(heatmap) > 0}")
except Exception as e:
    print(f"CNN Failed: {e}")

# Cleanup
if os.path.exists(dummy_img_path):
    os.remove(dummy_img_path)
