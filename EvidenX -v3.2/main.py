import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import secrets
import asyncio
from concurrent.futures import ThreadPoolExecutor

from engine.ela import perform_ela
from engine.metadata import extract_metadata
from engine.copymove import detect_copymove
from engine.cnn import predict_cnn, load_model
from engine.audio import analyze_audio
from engine.audio_lstm import load_audio_lstm_model
from engine.gan import load_gan_model
from engine.diffusion import load_diffusion_model

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
            
            # 1. Metadata
            metadata_task = loop.run_in_executor(executor, extract_metadata, filename)
            
            # 2. ELA
            ela_task = loop.run_in_executor(executor, perform_ela, filename)
            
            # 3. Copy-Move
            cm_task = loop.run_in_executor(executor, detect_copymove, filename)
            
            # 4. CNN (Deepfake Detection)
            cnn_task = loop.run_in_executor(executor, predict_cnn, filename)
            
            # Await all
            metadata_report, (ela_image, ela_score), (cm_count, cm_score), (cnn_score, cnn_heatmap) = await asyncio.gather(
                metadata_task, ela_task, cm_task, cnn_task
            )

            # --- Advanced Risk Scoring ---
            scores = {
                'ela': ela_score,
                'copymove': cm_score,
                'cnn': cnn_score,
                'metadata': 1.0 if metadata_report.get("Risk") == "High" else 0.0
            }
            
            # 1. Base max score (Strongest signal wins)
            max_metric = max(scores.values())
            
            # 2. Average of non-zero scores to add nuance/confidence
            non_zero_scores = [s for s in scores.values() if s > 0.1]
            avg_metric = sum(non_zero_scores) / len(non_zero_scores) if non_zero_scores else 0.0
            
            # 3. Final Weighted Score
            final_score = (max_metric * 0.7) + (avg_metric * 0.3)
            final_risk = min(max(final_score, 0.0), 1.0) * 100

            # Explainability Generation
            explanation = []
            if cnn_score > 0.6:
                explanation.append(f"AI Model detected deepfake patterns ({cnn_score*100:.1f}% confidence).")
            elif cnn_score > 0.4:
                explanation.append("Potential visual anomalies identified by AI.")
                
            if ela_score > 0.5:
                explanation.append(f"High compression inconsistencies ({ela_score*100:.1f}% noise variance).")
                
            if cm_count > 5:
                explanation.append(f"Repeated regions detected ({cm_count} matches).")
                
            if metadata_report.get("Risk") == "High":
                 explanation.append("Suspicious software usage in metadata.")

            # Advanced Model attribution
            if gan_discriminator is not None and cnn_score > 0.7:
                 explanation.append("GAN-specific artifacts potentially detected.")
            if diffusion_model is not None and cnn_score > 0.6:
                 explanation.append("Diffusion-based reconstruction anomalies found.")
        
        if not explanation:
            explanation.append("No significant anomalies detected.")

        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
        
        return JSONResponse(content={
            "filename": file.filename,
            "risk_score": round(final_risk, 1),
            "ela_image": ela_image,
            "ela_score": round(ela_score * 100, 1) if 'ela_score' in locals() else 0,
            "copymove_score": round(cm_score * 100, 1) if 'cm_score' in locals() else 0,
            "cnn_score": round(cnn_score * 100, 1) if 'cnn_score' in locals() else round(video_score*100, 1) if 'video_score' in locals() else 0,
            "cnn_heatmap": cnn_heatmap if 'cnn_heatmap' in locals() else None,
            "metadata": metadata_report,
            "explanation": explanation
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
                f"LSTM temporal analysis flagged {pct}% deepfake probability "
                "across spectrogram frames."
            )
            explanation.append(
                "Spectral irregularities and unnatural pitch transitions detected "
                "(high-attention regions highlighted in the attention map)."
            )
        elif score > 0.35:
            explanation.append(
                f"Moderate anomaly score ({pct}%). Some spectral features deviate "
                "from natural speech patterns."
            )
        else:
            explanation.append(
                f"Audio features appear natural (LSTM score: {pct}%)."
            )

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
        })

    except Exception as e:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
