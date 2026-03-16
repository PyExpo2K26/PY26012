# EvidenX v3.1 - Digital Forgery Detection System

## Overview
EvidenX is a state-of-the-art digital forensics platform designed to detect deepfakes and manipulated media (images, videos, and audio). It leverages advanced machine learning models, including MesoNet (CNN) for spatial artifacts and a Hybrid ResNeXt-LSTM model for temporal video analysis, alongside traditional forensic techniques like Error Level Analysis (ELA) and Metadata extraction.

## Key Features

### 1. Multi-Modal Detection
- **Image Forgery**: Detects manipulation using ELA, Copy-Move detection, and CNN-based deepfake classification.
- **Deepfake Video Analysis**: Uses a Hybrid ResNeXt + LSTM model to identify temporal inconsistencies in face sequences.
- **Audio Verification**: Spectral analysis to detect synthetic/AI-generated voice patterns.

### 2. Forensic Visualization
- **ELA Heatmaps**: Visualizes compression anomalies to highlight potential tampering.
- **CNN Grad-CAM**: Overlays heatmaps on images to show where the AI model detected manipulation.

### 3. Flexible Model Support
- **Pre-Trained Models**: Ready-to-use weights for immediate deepfake detection.
- **Custom Training**: Built-in scripts to train models on your own datasets.

### 4. Advanced Tech Stack
- **Backend**: Python, FastAPI, PyTorch, OpenCV.
- **Frontend**: Modern Cybernetic UI with real-time feedback.

---

## File Structure & Descriptions

### Core System
- **`main.py`**: The entry point of the application. Runs the FastAPI server and coordinates analysis pipelines.
- **`index.html`**: The frontend interface. Handles file uploads, visualization, and results display.
- **`requirements.txt`**: List of Python dependencies.

### Detection Engine (`engine/`)
- **`cnn.py`**: Implementation of the MesoNet model for image deepfake detection. Handles model loading (Custom > Pre-Trained > Random) and Grad-CAM generation.
- **`ela.py`**: Performs Error Level Analysis to detect compression artifacts.
- **`hybrid_model.py`**: Defines the ResNeXt + LSTM architecture for video analysis.
- **`video_utils.py`**: Utilities for face extraction from video frames.
- **`metadata.py` & `copymove.py`**: Modules for EXIF extraction and copy-move forgery detection.

### Utilities & Tools
- **`train_model.py`**: Script to train the MesoNet model on a custom dataset.
- **`download_weights.py`**: Script to download pre-trained MesoNet weights from the internet.
- **`debug_models.py`**: Diagnostic script to verify model loading and ELA functionality without running the full server.

---

## Installation & Setup

1. **Install Dependencies**:
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Models**:
   You have two options for the Deepfake Detection Model:
   
   *   **Option A: Use Pre-Trained Weights (Recommended for quick start)**
       Run the downloader script:
       ```bash
       python download_weights.py
       ```
       This fetches `mesonet_pre_trained.pth`.

   *   **Option B: Train on Your Own Data**
       See the "Training Custom Models" section below.

---

## Usage Instructions

### Running the Application
1. Start the server:
   ```bash
   python main.py
   ```
2. Open your browser and navigate to: `http://127.0.0.1:8000`
3. Upload an Image, Video, or Audio file to start analysis.

### Training Custom Models
If you want to train the model on a specific dataset:

1. Create a folder named `dataset` in the project root.
2. Inside `dataset`, create two subfolders: `real` and `fake`.
3. Place your training images into the respective folders.
4. Run the training script:
   ```bash
   python train_model.py
   ```
5. The script will save `mesonet_weights.pth`. Restart the application, and it will automatically prioritize these custom weights over pre-trained ones.

### Interpretation of Results
- **Risk Score**: Combined probability of forgery (0-100%).
    - **< 30%**: Likely Authentic.
    - **> 70%**: High Probability of Forgery.
- **ELA Heatmap**: Bright/Rainbow colored regions indicate areas that may have been digitally altered or have different compression levels than the rest of the image.
- **CNN Heatmap**: Red "hotspots" indicate regions where the AI features most strongly suggest a deepfake.

---
**EvidenX v3.1** - *Truth in the Digital Age*
