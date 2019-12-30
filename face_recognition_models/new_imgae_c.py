import pickle
import time

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential

pickle_in = open("Xx.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("yy.pickle", "rb")
y = pickle.load(pickle_in)

X = X / 255.0

dense_layers = [0]
layer_sizes = [64]
conv2D_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv2D_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            # for _ in range(dense_layer):
            # model.add(Dense(layer_size))
            model.add(Dense(12, input_dim=8))
            model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, y,
                      batch_size=30,
                      epochs=250,
                      # validation_split=0.3,
                      # validation_split=0.5,
                      callbacks=[tensorboard],
                      shuffle=True)

model.save('person_64x3-CNN.model')
