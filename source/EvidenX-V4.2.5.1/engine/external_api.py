import time
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"

def query_huggingface_api(file_path: str, api_key: str = None) -> dict:
    """
    Sends an image file to the Hugging Face Inference API for pin-accurate deepfake detection.
    Retries once on 503 Model Loading errors.
    """
    headers = {}
    if api_key:
         headers["Authorization"] = f"Bearer {api_key}"

    try:
        with open(file_path, "rb") as f:
            data = f.read()

        response = requests.post(HF_API_URL, headers=headers, data=data)
        
        # If the model is currently loading, HF returns 503. Wait and retry once.
        if response.status_code == 503:
            res_json = response.json()
            delay = res_json.get("estimated_time", 20.0)
            print(f"Hugging Face model loading. Waiting {delay}s...")
            time.sleep(delay + 1)
            response = requests.post(HF_API_URL, headers=headers, data=data)

        if response.status_code != 200:
            print(f"HF API Error {response.status_code}: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
            
        return {"result": response.json()}
    except Exception as e:
        print(f"External API exception: {e}")
        return {"error": str(e)}
