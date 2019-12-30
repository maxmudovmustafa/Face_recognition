import os
import numpy as np
import cv2
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath((__file__)))
img_dir = os.path.join(BASE_DIR, "image")
face_cascade = cv2.CascadeClassifier(
    "/home/warlock/Downloads/pych/my_project_faced/data/haarcascade_frontalface_default.xml")

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
            print id_
            orgimg = cv2.imread(path)
            #pil_image = Image.open(path).convert("L")
            img = cv2.resize(orgimg, (0, 0), fx=0.2, fy=0.2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('img', gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #image_aarray = np.array(pil_image, "uint8")

            #x_train.append(image_aarray)
            #y_labels.append(id_)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                minSize=(100, 100),
                minNeighbors=3)
            for (x, y, w, h) in faces:
                center = (x + w // 2, y + h // 2)
                radius = (w + h) // 4
                cv2.circle(img, center, radius, (255, 0, 0), 2)
                roi = gray[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

with open("/home/warlock/Downloads/pych/my_project_faced/face_recognition_models/pickles/labels.pickles", "wb") as f:
    pickle.dump(label_ids, f)
print label_ids
print np.array(y_labels)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("/home/warlock/Downloads/pych/my_project_faced/face_recognition_models/recognizers/trainner.yml")
