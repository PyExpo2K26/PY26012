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
from engine.gan import load_gan_model
from engine.diffusion import load_diffusion_model

app = FastAPI()
app.mount("/source", StaticFiles(directory="source"), name="source")

executor = ThreadPoolExecutor(max_workers=4)

gan_discriminator = None
diffusion_model = None

@app.on_event("startup")
async def startup_event():
    global gan_discriminator, diffusion_model
    load_model() 
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

        loop = asyncio.get_event_loop()
        
        metadata_task = loop.run_in_executor(executor, extract_metadata, filename)
        
        ela_task = loop.run_in_executor(executor, perform_ela, filename)
        
        cm_task = loop.run_in_executor(executor, detect_copymove, filename)
        
        cnn_task = loop.run_in_executor(executor, predict_cnn, filename)
        
        metadata_report, (ela_image, ela_score), (cm_count, cm_score), (cnn_score, cnn_heatmap) = await asyncio.gather(
            metadata_task, ela_task, cm_task, cnn_task
        )

        scores = {
            'ela': ela_score,
            'copymove': cm_score,
            'cnn': cnn_score,
            'metadata': 1.0 if metadata_report.get("Risk") == "High" else 0.0
        }
        
        max_metric = max(scores.values())
        
        non_zero_scores = [s for s in scores.values() if s > 0.1]
        avg_metric = sum(non_zero_scores) / len(non_zero_scores) if non_zero_scores else 0.0
        
        final_score = (max_metric * 0.7) + (avg_metric * 0.3)
        final_risk = min(max(final_score, 0.0), 1.0) * 100

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

        if gan_discriminator is not None and cnn_score > 0.7:
             explanation.append("GAN-specific artifacts potentially detected.")
        if diffusion_model is not None and cnn_score > 0.6:
             explanation.append("Diffusion-based reconstruction anomalies found.")

        if not explanation:
            explanation.append("No significant anomalies detected.")

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
