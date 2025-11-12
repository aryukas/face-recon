import sqlite3
import numpy as np
import io

DB_PATH = "face_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def add_face(name, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Store NumPy array as binary blob
    c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding.tobytes()))
    conn.commit()
    conn.close()

def get_all_faces():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM faces")
    rows = c.fetchall()
    conn.close()

    faces = []
    for name, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        faces.append({"name": name, "embedding": emb})
    return faces
