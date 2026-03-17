import sys
import os
import asyncio

# Patch sys.path because venv python.exe is broken
sys.path.insert(0, os.path.abspath('.venv/Lib/site-packages'))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_analyze():
    with open("test_image.jpg", "rb") as test_file:
        response = client.post("/analyze", files={"file": ("test_image.jpg", test_file, "image/jpeg")})
        
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Risk Score: {data.get('risk_score')}")
        print(f"ViT Score: {data.get('vit_score')}")
        print(f"CLIP Score: {data.get('clip_score')}")
        print("Explanation:")
        for exp in data.get('explanation', []):
            print(f" - {exp}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_analyze()
