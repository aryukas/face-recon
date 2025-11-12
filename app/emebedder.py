# app/embedder.py
import face_recognition
import numpy as np
from io import BytesIO
from PIL import Image

class FaceEmbedder:
    def __init__(self):
        pass

    def get_embeddings(self, image_bytes):
        """Return face encodings (128D embeddings)"""
        image = np.array(Image.open(BytesIO(image_bytes)))
        encodings = face_recognition.face_encodings(image)
        return encodings  # List of vectors
