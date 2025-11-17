import numpy as np
import faiss


class FaceMatcher:
    """
    FAISS-based face matcher with correct normalization and
    improved matching thresholds for FaceNet embeddings.
    """

    def __init__(self, embedding_dim: int = 512, use_cosine: bool = True):
        self.embedding_dim = embedding_dim
        self.use_cosine = use_cosine

        # Cosine = normalized vectors + Inner Product
        if use_cosine:
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)

        self.labels = []   # stores metadata

    # -------------------------------------------------------------
    # Helper to prepare an embedding (reshape + normalize)
    # -------------------------------------------------------------
    def _prepare(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)

        # Ensure 2D
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Validate dimension
        if emb.shape[1] != self.embedding_dim:
            raise ValueError(
                f"[FaceMatcher] Expected embedding size {self.embedding_dim}, got {emb.shape[1]}"
            )

        # Normalize for cosine similarity
        if self.use_cosine:
            faiss.normalize_L2(emb)

        return emb

    # -------------------------------------------------------------
    # Add face embedding to FAISS database
    # -------------------------------------------------------------
    def add_embedding(self, emb: np.ndarray, person_id: str, filename: str):
        emb = self._prepare(emb)
        self.index.add(emb)

        self.labels.append({
            "person_id": person_id,
            "filename": filename
        })

    # -------------------------------------------------------------
    # Core match function
    # -------------------------------------------------------------
    def match(self, emb: np.ndarray, threshold: float = 0.75):
        """
        Match a given face embedding against stored embeddings.

        Cosine similarity mode:
            similarity >= threshold → match

        L2 mode:
            distance <= threshold  → match
        """
        if len(self.labels) == 0:
            return None, None

        emb = self._prepare(emb)

        # Search for best match (top-1)
        scores, idxs = self.index.search(emb, 1)

        score = float(scores[0][0])
        idx = int(idxs[0][0])

        # -----------------------------
        # COSINE SIMILARITY MODE
        # -----------------------------
        if self.use_cosine:
            # Typical FaceNet good matches: 0.75–0.90
            if score >= threshold:
                return self.labels[idx], score
            return None, score

        # -----------------------------
        # EUCLIDEAN DISTANCE MODE
        # -----------------------------
        else:
            # L2 good matches usually < 1.1
            if score <= threshold:
                return self.labels[idx], score
            return None, score
