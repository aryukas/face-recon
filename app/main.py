from fastapi import FastAPI, UploadFile, File
from app.detector import FaceDetector
from app.embedder import FaceEmbedder
from app.matcher import FaceMatcher
from app.db import init_db, get_all_faces


app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
    description="Offline-ready Face Detection & Matching API using MTCNN + FaceNet",
)


detector = FaceDetector()
embedder = FaceEmbedder()
matcher = FaceMatcher()

# Initialize local DB storage
init_db()

# Load stored embeddings into FAISS
saved_faces = get_all_faces()

for item in saved_faces:
    matcher.add_embedding(
        emb=item["embedding"],
        person_id=item["name"],
        filename=item.get("filename", "database")
    )


@app.get("/")
def home():
    return {"message": "Face Recognition API is running !"}


@app.post("/detect/")
async def detect_face(file: UploadFile = File(...)):
    """
    Detect face bounding boxes from an uploaded image.
    """
    try:
        image_bytes = await file.read()
        boxes = detector.detect_faces(image_bytes)

        return {
            "status": "success" if boxes else "no face detected",
            "count": len(boxes),
            "boxes": boxes
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/match/")
async def match_face(file: UploadFile = File(...)):
    """
    Match uploaded face with database embeddings.
    """
    try:
        image_bytes = await file.read()

        # Extract embedding
        embedding = detector.get_embedding(image_bytes)
        if embedding is None:
            return {"status": "no face detected"}

        # Match with FAISS index
        match, distance = matcher.match(embedding)

        if match is None:
            return {"status": "no match", "distance": distance}

        return {
            "status": "match",
            "person_id": match["person_id"],
            "filename": match["filename"],
            "distance": distance,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
