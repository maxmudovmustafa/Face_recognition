import os
import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
from sympy import fwht

#frequency = 1000
#num_samples = 48000
num_samples = 1000
#sampling_rate = 48000.0
#amplitude = 16000

file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's.wav')
wav_file = wave.open(file, 'r')

data = wav_file.readframes(num_samples)
data = struct.unpack('{n}h'.format(n=num_samples), data)
# This will give us the frequency we want
#data_fft = np.fft.fft(data)
#frequencies = np.abs(data_fft)
_hanning = fwht(data)
#print ("Transform  : ", transform)

#print data_fft
#print frequencies
wav_file.close()

plt.subplot(2, 1, 1)

plt.plot(data[:300])

plt.title("Original audio wave")

plt.subplot(2, 1, 2)

plt.plot(_hanning)

#plt.title("Hadamard Transform")

plt.xlim(0, 1200)

plt.show()

