
import os
import sys

# Add the current directory to sys.path so we can import from engine
sys.path.append(os.getcwd())

from engine.metadata import extract_metadata

def test_metadata_extraction():
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        # Create a dummy image if it doesn't exist
        from PIL import Image
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(image_path)
        print(f"Created dummy {image_path}")

    print(f"Extracting metadata from {image_path}...")
    try:
        result = extract_metadata(image_path)
        print("Extraction successful.")
        import json
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Extraction failed: {e}")

if __name__ == "__main__":
    test_metadata_extraction()
