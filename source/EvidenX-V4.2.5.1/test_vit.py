import os
import sys

# Add the current directory to path so we can import engine
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engine.vit import predict_vit

def verify_vit():
    sample_image = "real_1017.jpg"  # Exists in Files folder based on list_dir
    if not os.path.exists(sample_image):
         # Try with full path or another file 
         files = [f for f in os.listdir(".") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
         if files:
              sample_image = files[0]
         else:
              print("ERROR: No image found to test.")
              return

    print(f"Testing ViT weight with image: {sample_image}")
    try:
         score = predict_vit(sample_image)
         print(f"SUCCESS: ViT score returned: {score}")
         if 0.0 <= score <= 1.0:
              print("Verification passed.")
         else:
              print("Verification failed: score out of bounds.")
    except Exception as e:
         print(f"Verification crashed: {e}")

if __name__ == "__main__":
    verify_vit()
