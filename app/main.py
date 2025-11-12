from fastapi import FastAPI, UploadFile, File
from app.detector import FaceDetector
from app.db import init_db

# Initialize FastAPI application
app = FastAPI(
    title="Face Recognition API",
    version="1.0",
    description="Simple and offline-ready face detection API using FastAPI and FaceNet.",
)

# Initialize modules once on startup
detector = FaceDetector()
init_db()


@app.get("/")
def home():
    """Check API status."""
    return {"message": "BSDK chalu hogaya"}


@app.post("/detect/")
async def detect_face(file: UploadFile = File(...)):
    """
    Detect faces from an uploaded image.

    Args:
        file: Image file (jpg/png/jpeg)

    Returns:
        JSON with total faces detected and operation status.
    """
    try:
        # Read the uploaded image as bytes
        image_bytes = await file.read()

        # Detect faces using MTCNN
        boxes = detector.detect_faces(image_bytes)

        # Build response
        return {
            "status": "success" if boxes else "no face detected",
            "faces_detected": len(boxes) if boxes else 0
        }

    except Exception as e:
        # Graceful error handling
        return {
            "status": "error",
            "message": f"Failed to process image: {str(e)}"
        }
