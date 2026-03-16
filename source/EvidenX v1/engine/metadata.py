from PIL import Image
from PIL.ExifTags import TAGS
import piexif

def extract_metadata(image_path):
    """
    Extracts EXIF metadata and checks for suspicious software signatures.
    """
    data_report = {
        "Software": "Unknown",
        "DateTime": "Unknown",
        "Model": "Unknown",
        "Risk": "Low"
    }
    
    try:
        image = Image.open(image_path)
        exif_raw = image._getexif()
        
        if not exif_raw:
