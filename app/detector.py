import io
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceDetector:
    """Unified detector + embedder for consistent face recognition."""

    def __init__(self):
        self.device = torch.device("cpu")

        # Use a fixed-size MTCNN for *consistent* alignment
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,     # return only best face
            device=self.device
        )

        # FaceNet (512-d embedding model)
        self.model = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(self.device)
        )

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """Convert uploaded image bytes to RGB PIL image."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img.convert("RGB")
        except Exception:
            return None

    def detect_faces(self, image_bytes: bytes):
        """Return bounding boxes for all detected faces."""
        img = self._load_image(image_bytes)
        if img is None:
            return []

        boxes, _ = self.mtcnn.detect(img)
        return boxes.tolist() if boxes is not None else []

    def get_embedding(self, image_bytes: bytes):
        """
        Generate normalized 512-d embedding from aligned face.
        Returns None if no face detected.
        """
        img = self._load_image(image_bytes)
        if img is None:
            return None

        # aligned: Tensor [3,160,160]
        aligned = self.mtcnn(img)
        if aligned is None:
            return None  # No face detected

        aligned = aligned.to(self.device).unsqueeze(0)  # Add batch dim

        # Generate FaceNet embedding
        with torch.no_grad():
            emb = self.model(aligned).cpu().numpy()[0]

        # Normalize embedding (critical for comparisons)
        emb = emb / np.linalg.norm(emb)

        return emb
