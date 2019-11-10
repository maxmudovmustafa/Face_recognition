import os

import cv2
import tensorflow as tf

CATEGORIES = ["Cat", "Dog"]


def prepare(file_path):
    IMAGE_SIZE = 100
    print path
    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    return new_array.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, 'p.jpg')

model = tf.keras.models.load_model("new_64x3-CNN.model")

prediction = model.predict(prepare(path))
print prediction
if (prediction[0][0] != 1.0 or prediction[0][0] != 0.0):
# print int(prediction[0][0])
    print CATEGORIES[int(prediction[0][0])]
else: print "unknow"
