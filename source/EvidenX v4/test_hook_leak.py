import sys
import os
sys.path.append(r"c:\Users\KiTE\Desktop\EvidenX v4.5.2\Files")

import torch
import cv2
import numpy as np
from engine.cnn import predict_cnn, load_model, model

def test_hook_leak():
    print("Loading model...")
    load_model()  # Initialize global model
    
    import engine.cnn as cnn_module
    m = cnn_module.model
    if m is None:
         print("Model load failed.")
         return
         
    target_layer = m.model.layer4
    
    # Check initial hooks
    print(f"Initial forward hooks: {len(target_layer._forward_hooks)}")
    print(f"Initial backward hooks: {len(target_layer._backward_hooks)}")
    
    # Create dummy image on disk for predict_cnn to read
    dummy_img_path = "temp_test_image.jpg"
    cv2.imwrite(dummy_img_path, np.zeros((256, 256, 3), dtype=np.uint8))
    
    print("\nCalling predict_cnn 3 times...")
    for i in range(3):
        predict_cnn(dummy_img_path)
        print(f"After run {i+1}:")
        print(f"  Forward hooks: {len(target_layer._forward_hooks)}")
        print(f"  Backward hooks: {len(target_layer._backward_hooks)}")
        
    # Cleanup
    if os.path.exists(dummy_img_path):
        os.remove(dummy_img_path)

if __name__ == "__main__":
    test_hook_leak()
