import requests
import os

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

if __name__ == "__main__":
    
    import torch
    from engine.cnn import Meso4

    print("Initializing MesoNet model structure...")
    model = Meso4()
    
    weights_path = "mesonet_weights.pth"
    if not os.path.exists(weights_path):
        print("No pre-trained weights found online for direct download.")
        print("Saving initialized weights to 'mesonet_weights.pth' to ensure application stability.")
        print("IMPORTANT: Replace this file with trained weights for real detection accuracy.")
        torch.save(model.state_dict(), weights_path)
    else:
        print(f"Weights file '{weights_path}' already exists.")
