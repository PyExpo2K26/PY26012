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

app = FastAPI()
app.mount("/source", StaticFiles(directory="source"), name="source")

# ThreadPool for CPU-bound tasks (CV2, PyTorch, Librosa)
executor = ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def startup_event():
    load_model()
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

        # Run analysis sections in parallel/threadpool for "Real-time" speedup
        loop = asyncio.get_event_loop()
        
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
        # Instead of simple weighted average, we use a Logic-based Scoring.
        # If ANY reliable detector (CNN, Metadata, ELA) is very high, the risk should be high.
        # We don't want a low Copy-Move score to drag down a high CNN detection.

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
        # 70% weight to the highest detector (Max Pooling concept), 30% to the average
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
        elif metadata_report.get("Risk") == "Medium":
             explanation.append("Missing or incomplete metadata.")

        if not explanation:
            explanation.append("No significant anomalies detected.")

        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
        
        return JSONResponse(content={
            "filename": file.filename,
            "risk_score": round(final_risk, 1),
            "ela_image": ela_image,
            "ela_score": round(ela_score * 100, 1),
            "copymove_score": round(cm_score * 100, 1),
            "cnn_score": round(cnn_score * 100, 1),
            "cnn_heatmap": cnn_heatmap,
            "metadata": metadata_report,
            "explanation": explanation
        })
        
    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/analyze_audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    try:
        filename = f"temp_audio_{secrets.token_hex(8)}_{file.filename}"
        with open(filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loop = asyncio.get_event_loop()
        score, spectrum_base64 = await loop.run_in_executor(executor, analyze_audio, filename)

        explanation = []
        if score > 0.5:
            explanation.append("High probability of synthetic audio generation (Smooth spectral features).")
        else:
            explanation.append("Audio features appear natural.")

        if os.path.exists(filename):
            os.remove(filename)

        return JSONResponse(content={
            "filename": file.filename,
            "fake_probability": round(score * 100, 1),
            "spectrum_image": spectrum_base64,
            "explanation": explanation
        })

    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
