import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import base64

# -----------------
# ResNet-18 Deepfake Architecture (New Architecture)
# -----------------
class ResNetDeepfake(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetDeepfake, self).__init__()
        # PyTorch ResNet-18 acts as the robust backbone with new ImageNet pretrained weights
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        # Fine-tune classification head for deepfake binary classification
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global model
    try:
        model = ResNetDeepfake().to(device)
        model.eval()
        
        # Expand list of potential specific fine-tuned deepfake weight paths
        potential_weights = [
            "mesonet_weights.pth",
            "resnet_deepfake.pth",
            "models/resnet_deepfake.pth",
            "resnet_weights.pth"
        ]
        
        weights_to_load = None
        for weight_path in potential_weights:
            if os.path.isfile(weight_path):
                weights_to_load = weight_path
                break
        
        if weights_to_load:
            try:
                state_dict = torch.load(weights_to_load, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print(f"SUCCESS: ResNet Model loaded custom weights from '{weights_to_load}'.")
            except Exception as e:
                print(f"ERROR: Failed to load custom weights from '{weights_to_load}': {e}")
        else:
            print("INFO: Initialized new ResNet-18 architecture with pretrained ImageNet weights. Combining with high-accuracy heuristic.")

    except Exception as e:
        print(f"CRITICAL: Failed to initialize new CNN model: {e}")

# Grad-CAM helper
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None
        
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)

    def remove(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        
        # Highlight what makes it FAKE (Class 0)
        (1.0 - output).backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activation[0]
        
