import os
import torch
import torchvision.transforms as transforms
from PIL import Image

model = None

def load_model():
    global model
    try:
        print("CNN Model placeholder initialized.")
        pass
    except Exception as e:
        print(f"Failed to load CNN model: {e}")

def predict_cnn(image_path):
    global model
    try:
       
        import numpy as np
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)
        variance = np.var(img_np)
        
        score = (variance % 100) / 100.0 
        
        return float(score)
        
    except Exception as e:
        print(f"CNN Prediction Error: {e}")
        return 0.5
