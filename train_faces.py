# train_faces.py
import os
import pickle
import face_recognition

def train_faces(input_folder="models/data/enroll", output_file="models/data/enroll/face_data.pkl"):
    encodings, names = [], []

    for person_name in os.listdir(input_folder):
        person_folder = os.path.join(input_folder, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_file in os.listdir(person_folder):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(person_folder, img_file)
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    encodings.append(encoding[0])
                    names.append(person_name)

    data = {"encodings": encodings, "names": names}
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Training complete. Saved to {output_file}")

if __name__ == "__main__":
    train_faces()
