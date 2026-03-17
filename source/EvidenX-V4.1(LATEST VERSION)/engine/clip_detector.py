import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# Using 'openai/clip-vit-base-patch32' for Zero-Shot Classification
MODEL_ID = "openai/clip-vit-base-patch32"

_processor = None
_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_clip_model():
    """
    Loads the CLIP model and processor from Hugging Face for zero-shot image classification.
    """
    global _processor, _model
    if _model is not None:
        return _processor, _model

    print(f"Loading CLIP model from {MODEL_ID}...")
    try:
        # Load processor and model
        _processor = CLIPProcessor.from_pretrained(MODEL_ID)
        _model = CLIPModel.from_pretrained(MODEL_ID)
        _model.to(device)
        _model.eval()
        print("SUCCESS: CLIP model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load CLIP model '{MODEL_ID}': {e}")
        _model = None
        _processor = None
        
    return _processor, _model

def predict_clip(image_path):
    """
    Predicts if the image is a deepfake using the CLIP model via zero-shot classification.
    Returns:
        float: Probability of being a DEEPFAKE [0.0, 1.0]
    """
    processor, model = load_clip_model()
    if model is None or processor is None:
        print("CLIP model not loaded. Skipping CLIP analysis.")
        return 0.0

    try:
         img = Image.open(image_path).convert('RGB')
         
         # Define text labels for zero-shot classification
         # [0]: Real/Authentic, [1]: Fake/Deepfake
         labels = ["a real authentic photograph", "an AI generated deepfake altered image"]
         
         # Preprocess image and text
         inputs = processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)
         
         with torch.no_grad():
             outputs = model(**inputs)
             # this is the image-text similarity score
             logits_per_image = outputs.logits_per_image 
             
         # Apply softmax to get probabilities
         probs = logits_per_image.softmax(dim=1)
         
         # Index 1 corresponds to "an AI generated deepfake altered image"
         fake_prob = probs[0][1].item()
         return float(fake_prob)
         
    except Exception as e:
         print(f"CLIP Prediction Error: {e}")
         return 0.0

if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        score = predict_clip(sys.argv[1])
        print(f"Image: {sys.argv[1]}")
        print(f"CLIP Deepfake Probability: {score:.4f}")
    else:
        print("Usage: python clip_detector.py <image_path>")
