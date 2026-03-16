import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import secrets
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from engine.ela import perform_ela
from engine.metadata import extract_metadata
from engine.copymove import detect_copymove
from engine.cnn import predict_cnn, load_model
from engine.audio import analyze_audio
from engine.audio_lstm import load_audio_lstm_model
from engine.gan import load_gan_model
from engine.diffusion import load_diffusion_model
from engine.email import analyze_email_headers
from engine.pcap import analyze_pcap
from engine.frequency import analyze_frequency
from engine.noise import analyze_noise
from engine.vit import predict_vit


app = FastAPI()
app.mount("/source", StaticFiles(directory="source"), name="source")

# ThreadPool for CPU-bound tasks (CV2, PyTorch, Librosa)
executor = ThreadPoolExecutor(max_workers=4)

# Global models
gan_discriminator = None
diffusion_model = None

@app.on_event("startup")
async def startup_event():
    global gan_discriminator, diffusion_model
    load_model()  # CNN
    try:
        load_audio_lstm_model()
        print("Audio LSTM loaded.")
    except Exception as e:
        print(f"Failed to load Audio LSTM: {e}")
    # Load advanced models
    try:
        gan_d, _ = load_gan_model()
        gan_discriminator = gan_d
        print("GAN Discriminator loaded.")
    except Exception as e:
        print(f"Failed to load GAN: {e}")
        
    try:
        diffusion_model = load_diffusion_model()
        print("Diffusion Model loaded.")
    except Exception as e:
        print(f"Failed to load Diffusion Model: {e}")

    # Ensure source directory exists
    os.makedirs("source", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        filename = f"temp_{secrets.token_hex(8)}_{file.filename}"
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Determine if video or image
        is_video = filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        # ThreadPool for CPU-bound tasks
        loop = asyncio.get_event_loop()

        if is_video:
            from engine.hybrid_model import predict_video
            from engine.video_utils import extract_faces
            import cv2
            
            # 1. Extract Faces (Key function for Deepfake Video)
            faces = await loop.run_in_executor(executor, extract_faces, filename, 10)
            
            # 2. Run Hybrid Model (ResNeXt + LSTM)
            video_score = await loop.run_in_executor(executor, predict_video, faces)
            
            # 3. For UI visualization, we need a representative image (Frame 0)
            # We'll save the first frame as a temp image to run the standard image tools on it
            # so the user sees ELA/Metadata for at least one frame.
            temp_frame_path = filename + "_frame0.jpg"
            cap = cv2.VideoCapture(filename)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(temp_frame_path, frame)
            cap.release()
            
            # Run Image Analysis on the single frame for UI completeness
            target_file_for_analysis = temp_frame_path if ret else filename 
            
            # Parallel Execution
            metadata_task = loop.run_in_executor(executor, extract_metadata, filename) # Video metadata
            ela_task = loop.run_in_executor(executor, perform_ela, target_file_for_analysis)
            cm_task = loop.run_in_executor(executor, detect_copymove, target_file_for_analysis)
            cnn_task = loop.run_in_executor(executor, predict_cnn, target_file_for_analysis)
            
            metadata_report, (ela_image, ela_score), (cm_count, cm_score), (cnn_score_frame, cnn_heatmap) = await asyncio.gather(
                metadata_task, ela_task, cm_task, cnn_task
            )
            
            # Cleanup temp frame
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)

            # Combined Score: Video Model is heavily weighted for videos
            final_risk_score = (video_score * 0.8) + (cnn_score_frame * 0.1) + (ela_score * 0.1)
            final_risk = min(max(final_risk_score, 0.0), 1.0) * 100
            
            explanation = [f"Deepfake Video Confidence: {video_score*100:.1f}%"]
            if video_score > 0.7:
                explanation.append("Temporal inconsistencies detected by LSTM.")
            if cnn_score_frame > 0.6:
                explanation.append("Spatial anomalies found in keyframes.")

        else:
            # Standard Image Analysis
            
            # Parallel Execution
            metadata_task = loop.run_in_executor(executor, extract_metadata, filename)
            ela_task      = loop.run_in_executor(executor, perform_ela, filename)
            cm_task       = loop.run_in_executor(executor, detect_copymove, filename)
            cnn_task      = loop.run_in_executor(executor, predict_cnn, filename)
            vit_task      = loop.run_in_executor(executor, predict_vit, filename)
            freq_task     = loop.run_in_executor(executor, analyze_frequency, filename)
            noise_task    = loop.run_in_executor(executor, analyze_noise, filename)

            # Await all
            (
                metadata_report,
                (ela_image, ela_score),
                (cm_count, cm_score, cm_heatmap),
                (cnn_score, cnn_heatmap),
                vit_score,
                (freq_score, fft_image, dct_image),
                (noise_score, noise_image),
            ) = await asyncio.gather(
                metadata_task, ela_task, cm_task, cnn_task, vit_task, freq_task, noise_task
            )

            # ── ViT Model Only Scoring ──────────────────────────
            final_weighted_score = vit_score
            final_risk = round(vit_score * 100, 1)
            
            # Since only one model is used, confidence is high for the single model prediction
            confidence_score = 98.0 if vit_score > 0.8 else 95.0
            
            # Explainability
            explanation = []
            if vit_score > 0.85:
                explanation.append(f"CRITICAL: High-confidence ViT V2 deepfake signature detected ({vit_score*100:.1f}% probability).")
                explanation.append("Vision Transformer analysis confirms synthetic pattern distribution.")
            elif vit_score > 0.6:
                explanation.append(f"ViT Model detected deepfake patterns ({vit_score*100:.1f}% confidence).")
            elif vit_score > 0.4:
                explanation.append("Potential visual anomalies identified by ViT analysis.")
        
        if not explanation:
            explanation.append("No significant anomalies detected.")

        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
        
        return JSONResponse(content={
            "filename":          file.filename,
            "risk_score":        round(final_risk, 1) if 'final_risk' in locals() else 0,
            "confidence_score":  confidence_score if 'confidence_score' in locals() else 80.0,
            "ela_image":         ela_image if 'ela_image' in locals() else None,
            "ela_score":         round(ela_score * 100, 1) if 'ela_score' in locals() else 0,
            "copymove_score":    round(cm_score * 100, 1) if 'cm_score' in locals() else 0,
            "copymove_heatmap":  cm_heatmap if 'cm_heatmap' in locals() else None,
            "vit_score":         round(vit_score * 100, 1) if 'vit_score' in locals() else 0,
            "cnn_score":         round(cnn_score * 100, 1) if 'cnn_score' in locals() else round(video_score * 100, 1) if 'video_score' in locals() else 0,
            "cnn_heatmap":       cnn_heatmap if 'cnn_heatmap' in locals() else None,
            "fft_image":         fft_image if 'fft_image' in locals() else None,
            "dct_image":         dct_image if 'dct_image' in locals() else None,
            "noise_image":       noise_image if 'noise_image' in locals() else None,
            "freq_score":        round(freq_score * 100, 1) if 'freq_score' in locals() else 0,
            "noise_score":       round(noise_score * 100, 1) if 'noise_score' in locals() else 0,
            "metadata":          metadata_report,
            "explanation":       explanation
        })
        
    except Exception as e:
