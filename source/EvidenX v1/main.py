import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import secrets
from engine.ela import perform_ela
from engine.metadata import extract_metadata
from engine.copymove import detect_copymove
from engine.cnn import predict_cnn, load_model

app = FastAPI()
app.mount("/source", StaticFiles(directory="source"), name="source")
@app.on_event("startup")
async def startup_event():
    load_model()

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
 

        metadata_report = extract_metadata(filename)

        ela_image, ela_score = perform_ela(filename)

        cm_count, cm_score = detect_copymove(filename)

        cnn_score = predict_cnn(filename)

        meta_score = 1.0 if metadata_report.get("Risk") == "High" else 0.0
        
        final_score = (ela_score * 0.3) + (cm_score * 0.2) + (cnn_score * 0.4) + (meta_score * 0.1)
        final_risk = min(max(final_score, 0.0), 1.0) * 100 # percentage

        os.remove(filename)
        
        return JSONResponse(content={
            "filename": file.filename,
            "risk_score": round(final_risk, 1),
            "ela_image": ela_image,
            "ela_score": round(ela_score * 100, 1),
            "copymove_score": round(cm_score * 100, 1),
            "cnn_score": round(cnn_score * 100, 1),
            "metadata": metadata_report
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
