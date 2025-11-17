import numpy as np
import faiss


class FaceMatcher:
    """
    Lightweight FAISS-based face matcher.
    Stores embeddings + person info and performs fast similarity search.
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim

        # Create FAISS index for L2 similarity
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Store metadata (person_id + filename)
        self.labels = []

    def _prepare(self, emb: np.ndarray):
        """
        Ensure correct shape and dtype.
        """
        emb = np.asarray(emb, dtype=np.float32)

        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        if emb.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding size mismatch: Expected {self.embedding_dim}, got {emb.shape[1]}"
            )

        return emb

    def add_embedding(self, emb: np.ndarray, person_id: str, filename: str):
        """
        Add a new face embedding to FAISS index.
        """
        emb = self._prepare(emb)
        self.index.add(emb)

        self.labels.append({
            "person_id": person_id,
            "filename": filename
        })

    def match(self, emb: np.ndarray, threshold: float = 0.8):
        """
        Match an embedding against all stored embeddings.

        Returns:
            (dict or None, float) → (Matched label, distance)
        """
        if len(self.labels) == 0:
            return None, None

        emb = self._prepare(emb)
        distances, indices = self.index.search(emb, 1)

        distance = float(distances[0][0])
        idx = int(indices[0][0])

        if distance < threshold:
            return self.labels[idx], distance

        return None, distance
