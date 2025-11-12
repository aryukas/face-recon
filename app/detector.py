import io
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceDetector:
    def __init__(self):
        # CPU-only
        device = torch.device("cpu")
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.device = device

    def detect_faces(self, image_bytes: bytes):
        """Return bounding boxes for detected faces"""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        boxes, _ = self.mtcnn.detect(img)
        return boxes.tolist() if boxes is not None else []

    def get_embedding(self, image_bytes: bytes):
        """Return 512-dim embedding for the first detected face"""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        aligned = self.mtcnn(img)
        if aligned is None:
            return None
        with torch.no_grad():
            embedding = self.resnet(aligned.to(self.device)).detach().cpu().numpy()
        return embedding[0]
