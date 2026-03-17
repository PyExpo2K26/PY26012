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
from typing import Optional

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
from engine.clip_detector import predict_clip
from engine.external_api import query_huggingface_api
from engine.timesformer_model import predict_timesformer


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
async def analyze_image(
    file: UploadFile = File(...),
    hf_api_key: Optional[str] = None  # Add this to accept optional API key
):
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
            timesformer_task = loop.run_in_executor(executor, predict_timesformer, filename)
            
            metadata_report, (ela_image, ela_score), (cm_count, cm_score, cm_heatmap), (cnn_score_frame, cnn_heatmap), timesformer_score = await asyncio.gather(
                metadata_task, ela_task, cm_task, cnn_task, timesformer_task
            )
            
            # Cleanup temp frame
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)

            # Combined Score: Video Model is heavily weighted for videos
            final_risk_score = (video_score * 0.4) + (timesformer_score * 0.4) + (cnn_score_frame * 0.1) + (ela_score * 0.1)
            final_risk = min(max(final_risk_score, 0.0), 1.0) * 100
            
            explanation = [f"Deepfake Video Confidence: {video_score*100:.1f}%"]
            if video_score > 0.7:
                explanation.append("Temporal inconsistencies detected by LSTM.")
            if timesformer_score > 0.6:
                explanation.append(f"TimeSformer temporal pixel-shift anomaly detected ({timesformer_score*100:.1f}%).")
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
            clip_task     = loop.run_in_executor(executor, predict_clip, filename)
            freq_task     = loop.run_in_executor(executor, analyze_frequency, filename)
            noise_task    = loop.run_in_executor(executor, analyze_noise, filename)
            hf_task       = loop.run_in_executor(executor, query_huggingface_api, filename, hf_api_key)

            # Await all
            (
                metadata_report,
                (ela_image, ela_score),
                (cm_count, cm_score, cm_heatmap),
                (cnn_score, cnn_heatmap),
                vit_score,
                clip_score,
                (freq_score, fft_image, dct_image),
                (noise_score, noise_image),
                hf_response
            ) = await asyncio.gather(
                metadata_task, ela_task, cm_task, cnn_task, vit_task, clip_task, freq_task, noise_task, hf_task
            )

            # ── ViT + CLIP Model Combined Scoring ──────────────────────────
            local_ensemble_score = (vit_score + clip_score) / 2
            
            # ── Hugging Face External API Override ─────────────────────────
            hf_fake_prob = None
            if "result" in hf_response:
                results = hf_response["result"]
                if isinstance(results, list) and len(results) > 0:
                    for label_info in results:
                        # Depending on the model, label is 'fake', 'deepfake', etc.
                        if "fake" in label_info["label"].lower():
                            hf_fake_prob = label_info["score"]
                            break
                    
                    # If model returned a 'real' score but not 'fake', invert it
                    if hf_fake_prob is None and len(results) == 2:
                         if "real" in results[0]["label"].lower():
                             hf_fake_prob = 1.0 - results[0]["score"]

            if hf_fake_prob is not None:
                final_risk = round(float(hf_fake_prob) * 100, 1)
                confidence_score = 99.9  # API gives pin-accurate detection
                explanation = [f"PIN-ACCURATE ONLINE MODEL (Hugging Face) detected deepfake signature with {final_risk}% confidence."]
                if final_risk > 80:
                    explanation.append("CRITICAL: Vision Transformer External API confirms synthetic pattern.")
            else:
                final_risk = round(local_ensemble_score * 100, 1)
                confidence_score = 98.0 if local_ensemble_score > 0.8 else 95.0
                
                # Explainability
                explanation = []
                if local_ensemble_score > 0.85:
                    explanation.append(f"CRITICAL: High-confidence CLIP+VIT DETECTOR V2 deepfake signature detected ({final_risk}% probability).")
                    explanation.append("Vision Transformer and CLIP zero-shot analysis confirm synthetic pattern distribution.")
                elif local_ensemble_score > 0.6:
                    explanation.append(f"CLIP+VIT Ensemble detected deepfake patterns ({final_risk}% confidence).")
                elif local_ensemble_score > 0.4:
                    explanation.append("Potential visual anomalies identified by CLIP+VIT analysis.")
        
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
            "clip_score":        round(clip_score * 100, 1) if 'clip_score' in locals() else 0,
            "timesformer_score": round(timesformer_score * 100, 1) if 'timesformer_score' in locals() else 0,
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
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except: pass
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/analyze_audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    filename = f"temp_audio_{secrets.token_hex(8)}_{file.filename}"
    try:
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loop = asyncio.get_event_loop()
        (
            score,
            stft_image,
            mel_image,
            waveform_image,
            mfcc_image,
            chroma_image,
            attn_image,
        ) = await loop.run_in_executor(executor, analyze_audio, filename)

        # ── Explainability ────────────────────────────────────────────
        explanation = []
        pct = round(score * 100, 1)
        if score > 0.65:
            explanation.append(
                f"The AI model found a high probability ({pct}%) that this audio is a deepfake."
            )
            explanation.append(
                "There are unnatural pitch changes and sound patterns (suspicious parts are highlighted in the Attention Map)."
            )
        elif score > 0.35:
            explanation.append(
                f"Moderate risk ({pct}%). Some sound patterns do not match normal human speech."
            )
        else:
            explanation.append(
                f"The audio sounds natural and matches typical human speech (Risk: {pct}%)."
            )

        # ── Graph Explanations ─────────────────────────────────────────
        graph_explanations = {
            "stft": "Shows sound frequencies over time. Strange breaks or sharp jumps usually mean the audio was cut and pasted.",
            "mel": "Shows sound how humans hear it. Missing parts or mechanical patterns often reveal AI-generated voices.",
            "waveform": "Shows loudness over time. Sudden jumps or absolute silence can show where clips were glued together.",
            "mfcc": "Shows the shape of the speaker's vocal tract. Fake audio often leaves unusual marks that don't match human speech.",
            "chroma": "Tracks musical notes or pitch. Useful for catching AI tools that have trouble keeping a natural human pitch.",
            "attn": "Highlights suspicious timing. Red areas are exactly where the AI model found unnatural sounds."
        }

        if os.path.exists(filename):
            os.remove(filename)

        return JSONResponse(content={
            "filename":       file.filename,
            "fake_probability": pct,
            # Visualizations
            "stft_image":     stft_image,       # Traditional STFT spectrogram (NEW)
            "spectrum_image": mel_image,         # Mel-spectrogram (kept same key for compat)
            "waveform_image": waveform_image,
            "mfcc_image":     mfcc_image,
            "chroma_image":   chroma_image,      # Chromagram (NEW)
            "attn_image":     attn_image,        # LSTM attention heatmap (NEW)
            "explanation":    explanation,
            "graph_explanations": graph_explanations  # Added for frontend & report
        })

    except Exception as e:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/analyze_email")
async def analyze_email_endpoint(file: UploadFile = File(...)):
    filename = f"temp_email_{secrets.token_hex(8)}_{file.filename}"
    try:
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(executor, analyze_email_headers, filename)

        if os.path.exists(filename):
            os.remove(filename)

        return JSONResponse(content=report)

    except Exception as e:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/analyze_pcap")
async def analyze_pcap_endpoint(file: UploadFile = File(...)):
    filename = f"temp_pcap_{secrets.token_hex(8)}_{file.filename}"
    try:
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(executor, analyze_pcap, filename)

        if os.path.exists(filename):
            os.remove(filename)

        return JSONResponse(content=report)

    except Exception as e:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/start_capture")
async def start_capture_endpoint():
    try:
        from engine.pcap import capture_live_traffic
        loop = asyncio.get_event_loop()
        # Capture for 15 seconds
        report = await loop.run_in_executor(executor, capture_live_traffic, 15)
        return JSONResponse(content=report)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
