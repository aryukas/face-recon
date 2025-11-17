import sqlite3
import numpy as np

DB_PATH = "face_data.db"


def init_db():
    """Initialize database and create table if not exists."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    embedding BLOB
                )
            """)
            conn.commit()
    except Exception as e:
        print(f"[DB ERROR] init_db failed: {e}")


def add_face(name, embedding):
    """Insert a face embedding (numpy array) into DB."""
    try:
        # Ensure clean dtype
        embedding = np.asarray(embedding, dtype=np.float32)

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO faces (name, embedding) VALUES (?, ?)",
                (name, embedding.tobytes())
            )
            conn.commit()
    except Exception as e:
        print(f"[DB ERROR] add_face failed: {e}")


def get_all_faces():
    """Load all embeddings from the DB and return list of dicts."""
    faces = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT name, embedding FROM faces")
            rows = c.fetchall()

        for name, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)

            # Ensure embedding shape is correct (1×512)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)

            faces.append({
                "name": name,
                "embedding": emb
            })

    except Exception as e:
        print(f"[DB ERROR] get_all_faces failed: {e}")

    return faces
