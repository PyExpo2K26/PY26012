import sys
import os
sys.path.append(r"c:\Users\KiTE\Desktop\EvidenX v4.5.2\Files")

import torch
from engine.cnn import ResNetDeepfake, GradCAM

def test_gradcam():
    print("Initializing model...")
    model = ResNetDeepfake()
    model.eval()
    
    device = torch.device("cpu")
    model.to(device)
    
    print("Setting up GradCAM...")
    target_layer = model.model.layer4
    grad_cam = GradCAM(model, target_layer)
    
    # Create dummy image tensor (Batch=1, C=3, H=256, W=256)
    img_t = torch.randn(1, 3, 256, 256, requires_grad=True)
    
    print("Running GradCAM...")
    try:
        heatmap, score = grad_cam(img_t)
        print(f"SUCCESS! Score: {score}")
        print(f"Heatmap shape: {heatmap.shape}")
    except Exception as e:
        print(f"ERROR: GradCAM failed: {e}")

if __name__ == "__main__":
    test_gradcam()
