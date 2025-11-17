from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.detector import FaceDetector
from app.embedder import FaceEmbedder
from app.matcher import FaceMatcher
from app.db import init_db, get_all_faces


app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
    description="Offline-ready Face Detection & Matching API using MTCNN + FaceNet",
)

# -------------------------
# CORS FIX – REQUIRED
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Initialize global modules
# -------------------------
detector = FaceDetector()
embedder = FaceEmbedder()
matcher = FaceMatcher()

# -------------------------
# Init DB + Load embeddings
# -------------------------
init_db()
saved_faces = get_all_faces()

for item in saved_faces:
    matcher.add_embedding(
        emb=item["embedding"],
        person_id=item["name"],
        filename=item.get("filename", "database")
    )

# -------------------------
# ROUTES
# -------------------------

@app.get("/")
def home():
    return {"message": "Face Recognition API is running!"}


@app.post("/detect/")
async def detect_face(file: UploadFile = File(...)):
    """
    Detect faces from an uploaded image.
    """
    try:
        image_bytes = await file.read()
        boxes = detector.detect_faces(image_bytes)

        return {
            "status": "success" if boxes else "no_face",
            "count": len(boxes),
            "boxes": boxes
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/match/")
async def match_face(file: UploadFile = File(...)):
    """
    Match uploaded image with stored face embeddings.
    """
    try:
        image_bytes = await file.read()

        # Extract embedding using detector + embedder
        embedding = detector.get_embedding(image_bytes)

        if embedding is None:
            return {"status": "no_face_detected"}

        # Compare with FAISS index
        match, distance = matcher.match(embedding)

        if match is None:
            return {
                "status": "no_match",
                "distance": float(distance)
            }

        return {
            "status": "match",
            "person_id": match["person_id"],
            "filename": match["filename"],
            "distance": float(distance),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
