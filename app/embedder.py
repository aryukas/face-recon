import io
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceEmbedder:

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # FIXED: pass device directly into MTCNN (do NOT use .to())
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            post_process=True,
            keep_all=False,
            device=self.device
        )

        # FaceNet model
        self.model = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(self.device)

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """Safely load image from bytes."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return img
        except Exception:
            return None

    def get_embeddings(self, image_bytes: bytes):
        """
        Returns:
            np.ndarray -> 512-dim embedding vector
            or
            None -> if no face detected
        """
        img = self._load_image(image_bytes)
        if img is None:
            return None   # invalid image

        # detect + align face
        aligned_face = self.mtcnn(img)

        if aligned_face is None:
            return None   # no face / detection failed

        aligned_face = aligned_face.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(aligned_face).cpu().numpy().reshape(-1)

        # FAISS requires float32
        emb = emb.astype(np.float32)

        return emb
