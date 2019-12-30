import os
import librosa   #for audio processing
#import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


path = os.path.join(BASE_DIR, 'gogo')
for audio in os.listdir(path):
    data_of_audio = os.path.join(path, audio)


warnings.filterwarnings("ignore")

labels = os.listdir(path)
no_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(path + '/' + label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))


# plot
plt.figure(figsize=(30, 5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each command')
#plt.show()

#labels = ["chapga", "ha", "o'ngga", "pastga", "tepaga", "yoq"]

duration_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples) / sample_rate))

plt.hist(np.array(duration_of_recordings))

all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        #if(len(samples)== 8000) :

        all_wave.append(samples)
all_label.append(label)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(all_label), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

from sklearn.model_selection import train_test_split
X = np.array(all_wave)
print y
#X= X.shape(1, 6, 48)
#X = X.reshape(X.shape[1:])
X = X.transpose()
print X
#y= y.shape(48,)
#X = X.reshape(X.shape[48:])
x_tr, x_val, y_tr, y_val = train_test_split(X,np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))
model.save('best_model.model')
model.save('best_model.hdf5')
print 'Hello'