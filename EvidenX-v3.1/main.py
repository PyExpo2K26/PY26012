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

# ThreadPool for CPU-bound tasks (CV2, PyTorch, Librosa)
executor = ThreadPoolExecutor(max_workers=4)

# Global models
gan_discriminator = None
diffusion_model = None

@app.on_event("startup")
async def startup_event():
    global gan_discriminator, diffusion_model
    load_model() # CNN
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
        
        # 5. Advanced Frameworks Analysis (GAN/Diffusion)
        # Note: Since these are CPU bound and we just want to prove they run, we'll simulate a check
        # In a real scenario, we'd run forward passes.
        # For this prototype, we'll assume if they loaded, they contribute a small weight.
        
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
        
