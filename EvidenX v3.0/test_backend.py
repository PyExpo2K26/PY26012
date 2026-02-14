import requests
import time
from PIL import Image
import os

img_path = "test_image.jpg"
if not os.path.exists(img_path):
    Image.new('RGB', (100, 100), color = 'red').save(img_path)

print("Waiting for server...")
time.sleep(3) 
try:
    print("Sending request...")
    with open(img_path, 'rb') as f:
        response = requests.post('http://127.0.0.1:8000/analyze', files={'file': f})
    
    if response.status_code == 200:
        print("Success! Response JSON:")
        print(response.json())
    else:
        print(f"Failed: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Connection failed: {e}")
