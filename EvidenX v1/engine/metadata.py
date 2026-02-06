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
            return data_report

        for tag_id, value in exif_raw.items():
            tag_name = TAGS.get(tag_id, tag_id)
            
            if tag_name == "Software":
                data_report["Software"] = str(value)
                # Heuristic check for editing software
                suspicious_keywords = ["Adobe", "Photoshop", "GIMP", "Picasa", "Paint", "Editor"]
                if any(keyword.lower() in str(value).lower() for keyword in suspicious_keywords):
                    data_report["Risk"] = "High"
            
            elif tag_name == "DateTime":
                data_report["DateTime"] = str(value)
            
            elif tag_name == "Model":
                data_report["Model"] = str(value)
                
        return data_report
        
    except Exception as e:
        print(f"Metadata Error: {e}")
        return data_report
