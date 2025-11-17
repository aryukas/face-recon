import io
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceDetector:
    """Lightweight Face Detection and Embedding Generator"""

    def __init__(self):
        # CPU-only usage for full offline support
        self.device = torch.device("cpu")

        # Initialize MTCNN for detection
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        # Load FaceNet model for embedding generation
        self.model = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(self.device)
        )

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """Safely convert byte data into an RGB PIL image"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img.convert("RGB")
        except Exception:
            return None  # Prevents crashes on corrupted images

    def detect_faces(self, image_bytes: bytes):
        """Return bounding boxes of detected faces"""
        img = self._load_image(image_bytes)
        if img is None:
            return []  # No valid image

        boxes, _ = self.mtcnn.detect(img)
        return boxes.tolist() if boxes is not None else []

    def get_embedding(self, image_bytes: bytes):
        """Generate a 512-dimensional embedding for the first detected face"""
        img = self._load_image(image_bytes)
        if img is None:
            return None

        aligned = self.mtcnn(img)
        if aligned is None:
            return None  # No face detected

        with torch.no_grad():
            embedding = self.model(aligned.to(self.device)).cpu().numpy()

        # Safely return first embedding
        return embedding[0] if embedding.size > 0 else None
