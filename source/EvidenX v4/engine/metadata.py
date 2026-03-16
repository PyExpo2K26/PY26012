from PIL import Image
from PIL.ExifTags import TAGS
import piexif
import subprocess
import json
import os

def get_exiftool_metadata(image_path):
    """
    Extracts metadata using ExifTool (if installed).
    Returns a dictionary of metadata or an error dict.
    """
    try:
        # Run exiftool and get JSON output
        result = subprocess.run(
            ["exiftool", "-json", image_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        metadata = json.loads(result.stdout)
        return metadata[0] if metadata else {}
    except FileNotFoundError:
        return {"error": "ExifTool not found in system PATH."}
    except subprocess.TimeoutExpired:
        return {"error": "ExifTool execution timed out."}
    except Exception as e:
        return {"error": f"ExifTool execution failed: {str(e)}"}

def get_imagemagick_metadata(image_path):
    """
    Extracts metadata using ImageMagick (magick identify).
    Returns a dictionary with 'raw' terminal output and 'attributes' dict.
    """
    try:
        # Run magick identify -verbose
        result = subprocess.run(
            ["magick", "identify", "-verbose", image_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        
        raw_output = result.stdout
        attributes = {}
        
        # Simple parsing for attributes (key: value)
        for line in raw_output.splitlines():
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                attributes[key.strip()] = val.strip()
                
        return {
            "raw": raw_output,
            "attributes": attributes
        }
    except FileNotFoundError:
        return {
            "raw": "ImageMagick (magick) not found in system PATH.",
            "attributes": {"error": "Tool not found"}
        }
    except subprocess.TimeoutExpired:
        return {
            "raw": "ImageMagick execution timed out.",
            "attributes": {"error": "Timeout"}
        }
    except Exception as e:
        return {
            "raw": f"ImageMagick execution failed: {str(e)}",
            "attributes": {"error": str(e)}
        }

def extract_metadata(image_path):
    """
    Extracts metadata using PIL, ExifTool, and ImageMagick.
    Checks for suspicious software signatures in PIL basic metadata.
    """
    basic_report = {
        "Software": "Unknown",
        "DateTime": "Unknown",
        "Model": "Unknown",
        "Risk": "Low"
    }
    
    try:
        image = Image.open(image_path)
        exif_raw = image._getexif()
        
        if exif_raw:
            for tag_id, value in exif_raw.items():
                tag_name = TAGS.get(tag_id, tag_id)
                
                if tag_name == "Software":
                    basic_report["Software"] = str(value)
                    # Heuristic check for editing software
                    suspicious_keywords = ["Adobe", "Photoshop", "GIMP", "Picasa", "Paint", "Editor"]
                    if any(keyword.lower() in str(value).lower() for keyword in suspicious_keywords):
                        basic_report["Risk"] = "High"
                
                elif tag_name == "DateTime":
                    basic_report["DateTime"] = str(value)
                
                elif tag_name == "Model":
                    basic_report["Model"] = str(value)
    except Exception as e:
        print(f"PIL Metadata Error: {e}")

    exiftool_data = get_exiftool_metadata(image_path)
    
    imagemagick_data = get_imagemagick_metadata(image_path)

    return {
        "basic": basic_report,
        "exiftool": exiftool_data,
        "imagemagick": imagemagick_data,
        "Risk": basic_report["Risk"] 
    }
