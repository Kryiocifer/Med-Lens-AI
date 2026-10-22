from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from app.models.inference import MedicalInference

app = FastAPI(title="MedLens AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference = MedicalInference(weights_dir="../weights")
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "MedLens AI Backend is running!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_scan(
    file: UploadFile = File(...),
    scan_type: str = Form("auto"),
):
    print(f"=== BACKEND RECEIVED scan_type: {scan_type} ===")

    if scan_type not in ["brain_tumor", "chest_xray", "auto"]:
        raise HTTPException(status_code=400, detail="Invalid scan_type")

    file_extension = file.filename.split(".")[-1]
    temp_filename = f"{uuid.uuid4()}.{file_extension}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = inference.predict(temp_path, scan_type)

        os.remove(temp_path)
        return JSONResponse(content=result)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_scan(
    file: UploadFile = File(...),
    scan_type: str = Form("brain_tumor"),
):
    # Grad-CAM only runs on a real model key
    if scan_type not in ["brain_tumor", "chest_xray"]:
        raise HTTPException(status_code=400, detail="Invalid scan_type")

    file_extension = file.filename.split(".")[-1]
    temp_filename = f"{uuid.uuid4()}.{file_extension}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        heatmap = inference.generate_gradcam(temp_path, scan_type)
        os.remove(temp_path)
        return JSONResponse(content={"heatmap": heatmap})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
