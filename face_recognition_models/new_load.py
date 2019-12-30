import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")
face_cascade = cv2.CascadeClassifier(
    "/home/warlock/Downloads/pych/my_project_faced/data/haarcascade_frontalface_alt2.xml")

print DATADIR
new_array = []
x_t = []
CATEGORIES = ["Johongir", "Kamoldin", "Salimjon", "shoxruxbek", "Umidjon"]
IMG_SIZE = 50
for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR, category)  # create path to dogs and cats
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
        img_array = np.array(img_array)
        faces = face_cascade.detectMultiScale(img_array, (IMG_SIZE, IMG_SIZE))

        for (x, y, w, h) in faces:
            roi = new_array[y:y + h, x:x + w]
            x_t.append(roi)

        plt.imshow(x_t, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  # ...and one more!

training_data = []
x_train = []


def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                print img
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                faces = face_cascade.detectMultiScale(new_array, scaleFactor=1.2, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = new_array[y:y + h, x:x + w]
                    x_train.append(roi)

                training_data.append([x_train, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            # except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            # except Exception as e:
            #    print("general exception", e, os.path.join(path,img))


create_training_data()

print(len(training_data))

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open("Xx.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("yy.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
