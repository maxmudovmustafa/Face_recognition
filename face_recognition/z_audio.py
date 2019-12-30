import os
import struct
import wave
from sympy import fwht
import numpy as np
from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's.wav')


waveFile = wave.open(BASE_DIR, 'r')
seq = []
length = waveFile.getnframes()
for i in range(0, length):
   waveData = waveFile.readframes(1)
   data = struct.unpack("<h", waveData)
   seq = data[0]
print seq
# import sympy


# sequence
seq = [23,
       56,
       12,
       555]

# hwht
transform = fwht(seq)
print ("Transform  : ", transform)

#
#sr, x = wavfile.read(BASE_DIR)

#x = signal.decimate(x, 4)
#x = x[48000*3:48000*3+8192]
#x *= np.hamming(8192)

#X = abs(np.fft.rfft(x))
#X_db = 20 * np.log10(X)
#freqs = np.fft.rfftfreq(8192, 1/48000)
#plt.plot(freqs, X_db)
#plt.show()