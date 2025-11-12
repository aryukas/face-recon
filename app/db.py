from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./face_recon.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
metadata = MetaData()

faces = Table(
    "faces",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("name", String, nullable=False),
    Column("embedding", LargeBinary, nullable=False)
)

SessionLocal = sessionmaker(bind=engine)

def init_db():
    metadata.create_all(engine)
