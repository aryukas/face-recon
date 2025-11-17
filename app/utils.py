import numpy as np
from PIL import Image
import io


def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Convert raw image bytes into a Pillow Image safely.
    """
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def to_numpy(tensor):
    """
    Convert tensor or list/array to float32 NumPy array.
    """
    if tensor is None:
        return None

    # Torch tensor → numpy
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    return arr.astype(np.float32)


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length.
    Improves matching consistency.
    """
    if emb is None:
        return None

    emb = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)

    if norm == 0:
        return emb

    return emb / norm


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute L2 distance between two embeddings.
    """
    if a is None or b is None:
        return float("inf")

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    return float(np.linalg.norm(a - b))


def resize_image(image: Image.Image, max_size: int = 800) -> Image.Image:
    """
    Resize large images while keeping aspect ratio.
    """
    if image is None:
        return None

    w, h = image.size

    if max(w, h) <= max_size:
        return image

    scale = max_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))

    return image.resize(new_size)
