from fastapi import FastAPI, UploadFile, File
from app.detector import FaceDetector
from app.db import init_db

app = FastAPI(title="Face Recon API", version="1.0")

# Initialize once on startup
detector = FaceDetector()
init_db()

@app.get("/")
def root():
    return {"message": "BSDK Face Recon API running!"}

@app.post("/detect/")
async def detect_face(file: UploadFile = File(...)):
    """Detect faces and return number of detected faces"""
    image_bytes = await file.read()
    boxes = detector.detect_faces(image_bytes)
    return {"faces_detected": len(boxes)}
