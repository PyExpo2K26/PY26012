import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Any

# Global model flag
model: Any = None
processor: Any = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_timesformer_model():
    global model, processor
    try:
        from transformers import AutoImageProcessor, TimesformerForVideoClassification
        # We can use the facebook/timesformer-base-finetuned-k400 or similar
        # Since it requires multiple frames, it's perfect for video anomaly detection
        model_name = "facebook/timesformer-base-finetuned-k400"
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = TimesformerForVideoClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        print("TimeSformer Video Classification Model Loaded.")
    except Exception as e:
        print(f"Error loading TimeSformer Model: {e}")

def predict_timesformer(video_path):
    """
    Extracts frames uniformly, passes them through TimeSformer.
    Returns an anomaly score between 0.0 and 1.0.
    """
    global model, processor
    if model is None or processor is None:
        load_timesformer_model()
    
    # Needs cv2
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.5 # Neutral fallback
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        return 0.5
        
    # TimeSformer expects exactly 8 frames by default for the base model
    num_frames = 8
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Replicate last frame if we run out early
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                break
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    cap.release()
    
    if len(frames) != num_frames:
        return 0.5 # Could not get enough frames
        
    try:
        inputs = processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Since K400 is kinetics, we use a proxy for "unnaturalness" or anomaly
            # A common approach without fine-tuning on deepfakes is evaluating prediction confidence entropy
            # or checking if action labels are scattered. 
            # Alternatively, if we just want it structured for future weights:
            score = 1.0 - float(probs.max().item()) # High entropy (uncertainty in action) => Anomaly
            
            # As a dummy metric while using K400 pre-trained
            return min(score * 2.0, 1.0) 
    except Exception as e:
        print(f"TimeSformer Inference Error: {e}")
        return 0.5
