import os
import numpy as np
import cv2
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath((__file__)))
img_dir = os.path.join(BASE_DIR, "image")
face_cascade = cv2.CascadeClassifier(
    "/home/warlock/Downloads/pych/my_project_faced/data/haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            # y_labels.append(label)
            # x_train.append(path)
            print label
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L")
            image_aarray = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_aarray, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_aarray[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)
print y_labels

with open("/home/warlock/Downloads/pych/my_project_faced/face_recognition_models/pickles/labels.pickles", "wb") as f:
    pickle.dump(label_ids, f)

print label_ids
recognizer.train(x_train, np.array(y_labels))
recognizer.save("/home/warlock/Downloads/pych/my_project_faced/face_recognition_models/recognizers/trainner.yml")
