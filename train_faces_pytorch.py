import os
import pickle
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

DATA_DIR = "models/data/enroll"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading models...")
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

encodings = []
names = []

print("[INFO] Starting face embedding generation...")

for person in os.listdir(DATA_DIR):
    person_path = os.path.join(DATA_DIR, person)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing: {person}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"[WARN] Could not open: {img_path}")
            continue

        face = mtcnn(img)

        if face is None:
            print(f"[WARN] No face detected in: {img_path}")
            continue

        with torch.no_grad():
            emb = resnet(face.unsqueeze(0)).cpu().numpy()[0]

        encodings.append(emb)
        names.append(person)

print("[INFO] Saving embeddings...")

output_path = "models/data/face_data.pkl"
with open(output_path, "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print(f"[SUCCESS] Training complete! Saved embeddings to {output_path}")
