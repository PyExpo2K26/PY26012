import requests
import json

url = 'http://127.0.0.1:8000/analyze'
files = {'file': ('test_video.mp4', open('c:/Users/KiTE/Desktop/EvidenX v4.5.2/Files/test_video.mp4', 'rb'), 'video/mp4')}

try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Response Text:", response.text)
except Exception as e:
    print("Error:", str(e))
