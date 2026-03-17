import os
import torch
from engine.cnn import load_model, predict_cnn

def run_test():
    print("\n--- Verifying Model Accuracy ---")
    load_model()
    
    images = ["real_1017.jpg", "fake_1011.jpg"]
    
    for img_path in images:
        if os.path.exists(img_path):
            print(f"\nTesting: {img_path}")
            try:
                score, _ = predict_cnn(img_path)
                print(f"RESULT: CNN Score (Fake Probability): {score:.4f}")
                if "real" in img_path:
                    if score < 0.5:
                        print("SUCCESS: Low score for Real image.")
                    else:
                        print("FAILURE: High score for Real image.")
                elif "fake" in img_path:
                    if score > 0.5:
                        print("SUCCESS: High score for Fake image.")
                    else:
                        print("FAILURE: Low score for Fake image.")
            except Exception as e:
                print(f"ERROR: Prediction failed: {e}")
        else:
            print(f"\nSkipping: {img_path} (File not found)")

if __name__ == "__main__":
    run_test()
