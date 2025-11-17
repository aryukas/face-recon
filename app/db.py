import sqlite3
import numpy as np

DB_PATH = "face_data.db"


def init_db():
    """
    Initialize SQLite DB.
    Creates table 'faces' if it does not exist.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            conn.commit()
    except Exception as e:
        print(f"[DB ERROR] init_db(): {e}")


def add_face(name: str, embedding: np.ndarray):
    """
    Save a single face embedding to DB.
    Embedding MUST be 512-dimensional and float32.
    """
    try:
        emb = np.asarray(embedding, dtype=np.float32)

        if emb.shape != (512,) and emb.shape != (1, 512):
            raise ValueError(
                f"Invalid embedding shape: {emb.shape} (expected 512 or 1×512)"
            )

        emb = emb.reshape(-1).astype(np.float32)

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO faces (name, embedding) VALUES (?, ?)",
                (name, emb.tobytes())
            )
            conn.commit()

    except Exception as e:
        print(f"[DB ERROR] add_face(): {e}")


def get_all_faces():
    """
    Load all faces from DB.

    Returns:
        list of {
            "name": str,
            "embedding": np.ndarray (1×512 float32)
        }
    """
    faces = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, embedding FROM faces")
            rows = cursor.fetchall()

        for name, emb_blob in rows:
            # Convert raw bytes → numpy
            emb = np.frombuffer(emb_blob, dtype=np.float32)

            # Ensure correct shape
            if emb.size != 512:
                print(f"[DB WARNING] Skipped corrupted embedding for {name}")
                continue

            faces.append({
                "name": name,
                "embedding": emb.reshape(1, 512)
            })

    except Exception as e:
        print(f"[DB ERROR] get_all_faces(): {e}")

    return faces
