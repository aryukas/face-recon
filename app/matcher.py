import numpy as np
import faiss


class FaceMatcher:
    def __init__(self, embedding_dim=512):
        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.labels = []

    def add_embedding(self, emb: np.ndarray, person_id: str, filename: str):
        """
        Add a new embedding to the database
        """
        emb = np.array(emb, dtype=np.float32).reshape(1, -1)
        self.index.add(emb)
        self.labels.append({
            "person_id": person_id,
            "filename": filename
        })

    def match(self, emb: np.ndarray, threshold: float = 0.8):
        """
        Compare given embedding with database and find closest match
        """
        if len(self.labels) == 0:
            return None, None

        emb = np.array(emb, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(emb, 1)
        distance = distances[0][0]
        match_index = indices[0][0]

        if distance < threshold:
            return self.labels[match_index], float(distance)
        return None, float(distance)
