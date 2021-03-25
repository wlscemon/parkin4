from pyrpde import  rpde
from scipy.io.wavfile import read
import numpy as np
import os

# make sure your audio data is in float32. Else, either use librosa or 
# normalize it to [-1,1] by dividing it by 2 ** 16 if it's 16bit PCM


# basepath = 'D:/data/PKS/NeuroSpeech-master/src/wavtest/'
# name = 'example2.wav'

# wavfile = basepath + name

def readname(filePath):
    name = os.listdir(filePath)
    return name

def writewavfeat(path,wavfiles,txt):
    for wavfile in wavfiles:

        audio = path +'/' + wavfile

        fs,data_audio = read(audio)

        entropy = rpde(data_audio, tau=30, dim=4, epsilon=0.01, tmax=1500)
    
        print('Done!!'+wavfile+'\n')

    
        with open(txt, 'a') as file:
            file.write(wavfile+','+str(entropy)+'\n')



# pathlist ='D:/data/PKS/NeuroSpeech-master/src/wav/' 
# for i in range(1,5):
#     path = pathlist+str(i)+'/raw'
#     name1 = readname(path)
#     writewavfeat(path, name1)

pathlist ='D:/data/PKS/NeuroSpeech-master/src/wavtest/' 
path = pathlist + '4'



name3 = readname(path)  #获得path路径下 文件名所组成的list
writewavfeat(path, name3, 'rpde.txt') #RPDE分析并写入txt



# wavfile = ['NC_003_1_2.wav']
# path = 'D:/data/PKS/NeuroSpeech-master/src/wavtest'

# writewavfeat(path,wavfile)