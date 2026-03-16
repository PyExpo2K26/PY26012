import cv2
import torch
import numpy as np
from PIL import Image

# Global MTCNN holder
mtcnn = None

def load_mtcnn():
    global mtcnn
    try:
        from facenet_pytorch import MTCNN
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=True, device=device)
        print("MTCNN Face Detection Loaded.")
    except ImportError:
        print("facenet-pytorch not found. Falling back to OpenCV.")
        mtcnn = "opencv"
    except Exception as e:
        print(f"Error loading MTCNN: {e}. Falling back to OpenCV.")
        mtcnn = "opencv"

def extract_faces(video_path, num_frames=10):
    """
    Extracts faces from a video.
    Returns:
        List of face tensors (or images) ready for model input.
        If no faces found, returns empty list.
    """
    global mtcnn
    if mtcnn is None:
        load_mtcnn()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0: return []
    
    # Calculate indices to sample frames uniformly
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    faces_batch = []
    
    current_frame = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        if mtcnn != "opencv":
            try:
                # MTCNN detection
                # We only take the largest face for single-person deepfake scenarios usually
                # But let's return all for robustness or strictly the best confidence.
                # For this implementation, let's keep it simple: assume 1 target face.
                
                boxes, _ = mtcnn.detect(pil_img)
                if boxes is not None:
                     # Crop the first face
                     box = boxes[0]
                     face = pil_img.crop(box)
                     faces_batch.append(face)
            except Exception as e:
                # print(f"MTCNN Inference Error: {e}")
                pass
        else:
            # OpenCV Fallback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = pil_img.crop((x, y, x+w, y+h))
                faces_batch.append(face)

    cap.release()
    return faces_batch
