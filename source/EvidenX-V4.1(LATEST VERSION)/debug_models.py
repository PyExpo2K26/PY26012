import os
import torch
import cv2
import numpy as np
from engine.ela import perform_ela
from engine.cnn import load_model, predict_cnn, model as global_model

def test_ela():
    print("\n--- Testing ELA ---")
    image_path = "fake_1011.jpg"
    if not os.path.exists(image_path):
        print(f"Test image {image_path} not found. Creating dummy.")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.putText(img, "Fake", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(image_path, img)

    try:
        ela_img, score = perform_ela(image_path)
        print(f"ELA Score: {score}")
        print(f"ELA Image Base64 len: {len(ela_img) if ela_img else 0}")
        if ela_img:
            print("ELA success.")
        else:
            print("ELA returned None image.")
    except Exception as e:
        print(f"ELA Failed: {e}")

def test_cnn():
    print("\n--- Testing CNN Model ---")
    load_model()
    
    # Access the global model from the module
    from engine.cnn import model
    
    if model is None:
        print("Model failed to load/initialize.")
        return

    # Check if weights are random or loaded
    # specific check: see if file existed
    if os.path.exists("mesonet_weights.pth"):
        print("STATUS: mesonet_weights.pth exists.", flush=True)
        try:
            state_dict = torch.load("mesonet_weights.pth", map_location=torch.device('cpu'))
            print("STATUS: Weights file loaded successfully with torch.load.", flush=True)
            # check a key
            k = list(state_dict.keys())[0]
            print(f"STATUS: First layer weights mean: {state_dict[k].float().mean().item()}", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to load weights file with torch: {e}", flush=True)
    else:
        print("WARNING: mesonet_weights.pth DOES NOT EXIST. Model is using random weights.", flush=True)

    image_path = "real_1017.jpg"
    if not os.path.exists(image_path):
         image_path = "fake_1011.jpg"

    if os.path.exists(image_path):
        print(f"STATUS: Testing prediction on {image_path}", flush=True)
        try:
            score, heatmap = predict_cnn(image_path)
            print(f"RESULT: CNN Score: {score}", flush=True)
            print(f"RESULT: Heatmap present: {heatmap is not None and len(heatmap) > 0}", flush=True)
        except Exception as e:
            print(f"ERROR: CNN Prediction failed: {e}", flush=True)
    else:
        print("WARNING: No test image found for CNN.", flush=True)

if __name__ == "__main__":
    test_ela()
    test_cnn()
