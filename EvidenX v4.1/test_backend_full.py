import requests
import cv2
import numpy as np
import os

BASE_URL = "http://127.0.0.1:8000"

def create_dummy_video(filename="test_video.avi"):
    # Create a 1-second dummy video (black frames)
    height, width = 240, 320
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for _ in range(fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add a "face" rectangle to ensure face detection might pick something up (or fail gracefully)
        cv2.rectangle(frame, (100, 50), (200, 150), (255, 255, 255), -1) 
        video.write(frame)
        
    video.release()
    return filename

def test_image_analysis():
    print("Testing Image Analysis...")
    # Create dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite("test_img_api.jpg", img)
    
    with open("test_img_api.jpg", "rb") as f:
        files = {"file": ("test_img_api.jpg", f, "image/jpeg")}
        try:
            response = requests.post(f"{BASE_URL}/analyze", files=files)
            if response.status_code == 200:
                print("Image Analysis: SUCCESS")
                print(response.json().keys())
            else:
                print(f"Image Analysis: FAILED ({response.status_code})")
                print(response.text)
        except Exception as e:
            print(f"Connection Error: {e}")

    os.remove("test_img_api.jpg")

def test_video_analysis():
    print("\nTesting Video Analysis...")
    vid_name = create_dummy_video()
    
    with open(vid_name, "rb") as f:
        files = {"file": (vid_name, f, "video/x-msvideo")}
        try:
            response = requests.post(f"{BASE_URL}/analyze", files=files, timeout=60)
            if response.status_code == 200:
                data = response.json()
                print("Video Analysis: SUCCESS")
                print(f"Risk Score: {data.get('risk_score')}")
                if data.get('cnn_score') == 0 and data.get('risk_score') > 0:
                     print("Note: cnn_score might be 0 if video model dominates or lacks weights, but risk_score is present.")
            else:
                print(f"Video Analysis: FAILED ({response.status_code})")
                print(response.text)
        except Exception as e:
            print(f"Connection Error: {e}")
            
    os.remove(vid_name)

if __name__ == "__main__":
    test_image_analysis()
    test_video_analysis()
