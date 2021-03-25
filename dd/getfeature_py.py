from pyrpde import rpde
# pip install pyrpde
from scipy.io import wavfile
import sys
sys.path.append('NeuroSpeech-master/src/prosody/')
import prosody
sys.path.append('NeuroSpeech-master/src/phonVowels/')
import phonVowels
sys.path.append('NeuroSpeech-master/src/DDK/')
import DDK

import numpy as np


def getfeature(audio_path):

    # RPDE方法必须输入float32格式
    fs, data_audio = wavfile.read(audio_path)

    # PRDE = rpde(data_audio, tau=30, dim=4, epsilon=0.01, tmax=1500)
    #RPDE = rpde(data_audio, tau=30, dim=4, epsilon=0.01, tmax=1500)
    # try:
    #     RPDE = rpde(audio_path, tau=30, dim=4, epsilon=0.01, tmax=1500)
    # except    Exception:
    #     pass


    path_base = 'NeuroSpeech-master/src/prosody/'
    F0, logE, mF0, sF0, mmF0, mlogE, slogE, mmlogE, Vrate, avgdurv, stddurv, Silrate, avgdurs, stddurs, F0varsemi = prosody.prosody(audio_path, path_base)

    path_base = 'NeuroSpeech-master/src/phonVowels/'
    F0, F0semi, mjitter, mshimmer, apq, ppq, mlogE, mcd, degreeU, varF0  = phonVowels.phonationVowels( audio_path, path_base )



    path_base = 'NeuroSpeech-master/src/DDK/'
    F0, logE, mF0, sF0, mmF0, mlogE, slogE, mmlogE, DDKrate, avgdurddk, stddurddk, Silrate, avgdurs, stddurs, F0varsemi=DDK.DDK(audio_path, path_base)

    if np.isnan(avgdurs):
        avgdurs=0
    if np.isnan(stddurs):
        stddurs=0


    f_prosofy = [F0, logE, mF0, sF0, mmF0, mlogE, slogE, mmlogE, Vrate, avgdurv, stddurv, Silrate, avgdurs, stddurs, F0varsemi]
    f_phon = [F0, F0semi, mjitter, mshimmer, apq, ppq, mlogE, mcd, degreeU, varF0]
    f_ddk = [F0, logE, mF0, sF0, mmF0, mlogE, slogE, mmlogE, DDKrate, avgdurddk, stddurddk, Silrate, avgdurs, stddurs,
             F0varsemi]
    f = []
    # if 'RPDE' in globals():
    #     f.extend(RPDE)
    # f.append(RPDE)
    f.extend(f_prosofy[2:])
    f.extend(f_phon[2:-3])
    f.extend(f_phon[-2:])
    f.extend(f_ddk[-3:-1])



    return f#我不知道罗正潮用了哪些量，python的都在上面