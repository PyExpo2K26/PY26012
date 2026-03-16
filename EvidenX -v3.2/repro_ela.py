
import os
import sys
import base64

# Add the current directory to sys.path
sys.path.append(os.getcwd())

from engine.ela import perform_ela

def test_ela_generation():
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        # Create a dummy image
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(image_path, img)
        print(f"Created dummy {image_path}")

    print(f"Generating ELA for {image_path}...")
    try:
        ela_image, score = perform_ela(image_path)
        
        if ela_image:
            print(f"ELA Generation Successful.")
            print(f"Score: {score}")
            print(f"Base64 Length: {len(ela_image)}")
            
            # Optionally save the output to verify visual
            with open("debug_ela_output.png", "wb") as f:
                f.write(base64.b64decode(ela_image))
            print("Saved debug_ela_output.png for manual inspection.")
        else:
            print("ELA Generation Failed (returned None).")
            
    except Exception as e:
        print(f"ELA Exception: {e}")

if __name__ == "__main__":
    test_ela_generation()
