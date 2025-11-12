import io
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceDetector:
    """Lightweight Face Detection and Embedding Generator"""

    def __init__(self):
        # Runs completely on CPU for offline use
        self.device = torch.device("cpu")

        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        # Load pretrained FaceNet model for feature extraction
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """Convert byte data into a PIL image"""
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def detect_faces(self, image_bytes: bytes):
        """Return bounding boxes of detected faces"""
        img = self._load_image(image_bytes)
        boxes, _ = self.mtcnn.detect(img)
        return boxes.tolist() if boxes is not None else []

    def get_embedding(self, image_bytes: bytes):
        """Generate a 512-dimensional embedding for the first detected face"""
        img = self._load_image(image_bytes)
        aligned = self.mtcnn(img)

        if aligned is None:
            return None

        with torch.no_grad():
            embedding = self.model(aligned.to(self.device)).cpu().numpy()
        return embedding[0]  # Return first face’s embedding
