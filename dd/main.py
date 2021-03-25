import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import wave, struct, math, random
from scipy.interpolate import CubicSpline  # for warping
from transforms3d.axangles import axangle2mat  # for rotation
from da.Example_DataAugmentation_TimeseriesData import *
from scipy.io.wavfile import write


y, sr = librosa.load('D:\datas\parkinson/1/raw/NC_001_1_1.wav', sr=None)
f = wave.open(r'D:\datas\parkinson/1/raw/NC_001_1_1.wav', "rb")

params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]

print(params)
# print(type(y))
# print(type(sr))
# print(len(y))
# print(sr)
# plt.figure()
# librosa.display.waveplot(y, sr)
# plt.title('lanarde')
# plt.show()
#
#
# myX = np.load('da/X_sample.npy')
print(y.shape)  ## 3600 samples (Time) * 3 features (X,Y,Z)
print(max(y))
print(min(y))
y = y[:sr]
ax1 = plt.subplot(1,2,1)
ax1.plot(y)
plt.title("An example of 1min acceleration data")
ax1.axis([0, sr, -0.3, 0.3])


sigma = 0.05


def DA_Jitter(X, sigma=0.001):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise
def DA_Scaling(X, sigma=0.01):

    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,1))# shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

ax2 = plt.subplot(1,2,2)
yy = DA_RandSampling(y)
ax2.plot(yy)
# file = os.path.join('da',)
# file.export(out_f = "louder_wav_file.wav",
#                        format = "wav")
# Wave_write.writeframesraw(yy)
# fig = plt.figure(figsize=(15, 4))
# for ii in range(8):
#     ax = fig.add_subplot(2, 4, ii + 1)
#     ax.plot(DA_Jitter(myX, sigma))
#     ax.set_xlim([0, 3600])
#     ax.set_ylim([-1.5, 1.5])

plt.show()

#
sampleRate = framerate # hertz
# duration = 1.0 # seconds
# frequency = 440.0 # hertz
obj = wave.open('soundfgf.wav','w')
obj.setnchannels(nchannels) # mono
obj.setsampwidth(sampwidth)
obj.setframerate(sampleRate)
# for i in range(99999):
#  value = random.randint(-32767, 32767)
#  data = struct.pack('<h', value)
obj.writeframesraw( yy )
obj.close()