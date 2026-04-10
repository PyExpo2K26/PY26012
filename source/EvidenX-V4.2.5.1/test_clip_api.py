import sys
import os

# Patch sys.path because venv python.exe is broken
sys.path.insert(0, os.path.abspath('.venv/Lib/site-packages'))

try:
    from engine.clip_detector import predict_clip
    score = predict_clip('test_image.jpg')
    print(f"CLIP Deepfake Probability for test_image.jpg: {score:.4f}")
except Exception as e:
    print(f"Test failed: {e}")
