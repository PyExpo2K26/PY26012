import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
# Model ID from Hugging Face for Deepfake Detection
# Using 'prithivMLmods/Deep-Fake-Detector-v2-Model' as researched
MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"

_processor = None
_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vit_model():
    """
    Loads the Vision Transformer (ViT) model and processor from Hugging Face.
    """
    global _processor, _model
    if _model is not None:
        return _processor, _model

    print(f"Loading ViT Detector V2 model from {MODEL_ID}...")
    try:
        # Load processor and model
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        _model.to(device)
        _model.eval()
        print("SUCCESS: ViT Detector V2 model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load ViT model '{MODEL_ID}': {e}")
        # Fallback list or dummy initialization can be added here if needed
        # For now, we raise or allow it to be None so predict handles it
        _model = None
        _processor = None
        
    return _processor, _model

def predict_vit(image_path):
    """
    Predicts if the image is a deepfake using the ViT model.
    Returns:
        float: Probability of being a DEEPFAKE [0.0, 1.0]
    """
    processor, model = load_vit_model()
    if model is None or processor is None:
        print("ViT model not loaded. Skipping ViT analysis.")
        return 0.0

    try:
         img = Image.open(image_path).convert('RGB')
         
         # Preprocess image
         inputs = processor(images=img, return_tensors="pt").to(device)
         
         with torch.no_grad():
             outputs = model(**inputs)
             logits = outputs.logits
             
         probs = torch.softmax(logits, dim=-1)
         
         # The model output indices map to labels.
         # For 'prithivMLmods/Deep-Fake-Detector-v2-Model':
         # Label 0 or 1 might be 'Deepfake' or 'Realism'.
         # We can read the config to be safe, or inspect model.config.id2label
         
         id2label = model.config.id2label
         fake_index = None
         
         # Find index for 'Deepfake' or 'fake'
         for idx, label in id2label.items():
              if 'fake' in label.lower() or 'deepfake' in label.lower():
                   fake_index = idx
                   break
                   
         if fake_index is None:
              # Fallback assumption if label names differ
              # Usually index 0 or 1. Let's assume index 1 if not found.
              print("WARNING: 'Deepfake' label not found in model config. Assuming index 1.")
              fake_index = 1
              
         fake_prob = probs[0][fake_index].item()
         return float(fake_prob)
         
    except Exception as e:
         print(f"ViT Prediction Error: {e}")
         return 0.0

if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        score = predict_vit(sys.argv[1])
        print(f"Image: {sys.argv[1]}")
        print(f"ViT Deepfake Probability: {score:.4f}")
    else:
        print("Usage: python vit.py <image_path>")
