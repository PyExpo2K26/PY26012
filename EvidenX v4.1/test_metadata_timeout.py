
import unittest
from unittest.mock import patch
import subprocess
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from engine.metadata import get_exiftool_metadata, get_imagemagick_metadata

class TestMetadataTimeout(unittest.TestCase):
    
    @patch('subprocess.run')
    def test_exiftool_timeout(self, mock_run):
        # Configure the mock to raise TimeoutExpired
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=['exiftool'], timeout=10)
        
        result = get_exiftool_metadata("dummy.jpg")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "ExifTool execution timed out.")
        print("ExifTool timeout handled correctly.")

    @patch('subprocess.run')
    def test_imagemagick_timeout(self, mock_run):
        # Configure the mock to raise TimeoutExpired
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=['magick'], timeout=10)
        
        result = get_imagemagick_metadata("dummy.jpg")
        
        self.assertIn("attributes", result)
        self.assertIn("error", result["attributes"])
        self.assertEqual(result["attributes"]["error"], "Timeout")
        print("ImageMagick timeout handled correctly.")

if __name__ == '__main__':
    unittest.main()
