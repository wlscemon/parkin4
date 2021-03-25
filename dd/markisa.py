from os import listdir
import os
from os.path import isfile, join
import librosa.display
import wave
import numpy as np
from da.Example_DataAugmentation_TimeseriesData import *
import matplotlib.pyplot as plt

video_path = 'D:/datas/parkinson'
content_list = [f for f in listdir(video_path) if not isfile(join(video_path, f))]
ag_list = ['jittering', 'scaling', 'mag_warping', 'time_warping', 'rotation', 'permutation', 'random_sampling',
           'combinations']


def new_ag_file(i, j):
    cur_dir = join(video_path, content_list[i])
    folder_name = ag_list[j]
    if os.path.isdir(cur_dir):
        os.mkdir(os.path.join(cur_dir, folder_name))


# new ag files
# for i in range(len(content_list)):
#     for j in range(len(ag_list)):
#         new_ag_file(i, j)


def get_ag_video(v, k):
    ag_v = DA_Jitter(v, sigma=0.05)
    if k == 1 :
        ag_v = DA_Jitter(v, sigma=0.05)
        pass
    elif k == 2:
        # ag_v = DA_Scaling(v, sigma=0.1)
        pass
    elif k ==3:
        ag_v = DA_MagWarp(v, sigma=0.2)
        pass
    elif k == 4:
        ag_v = DA_TimeWarp(v, sigma=0.05)
        pass
    elif k == 5:
        # ag_v = DA_Rotation(v)
        pass
    elif k == 6:
        # ag_v = DA_Permutation(v, nPerm=4, minSegLength=10)
        pass
    else:
        ag_v = DA_RandSampling(v)
        pass
    # else :
    #     ag_v = DA_Rotation(DA_Permutation(v, nPerm=4))
    #     pass
    return ag_v


# def DA_Jitter(X, sigma=0.001):
#     myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
#     return X + myNoise


for i in range(len(content_list)):
    c_path = join(video_path, content_list[i])
    # video_raw_list = [f for f in listdir(c_path) if not isfile(join(video_path, f))]
    v_path = join(c_path, 'raw')
    raw_list = [f for f in listdir(v_path) if isfile(join(v_path, f))]
    for j in range(len(raw_list)):
        raw_path = join(v_path, raw_list[j])
        # print(raw_path)
        # print(raw_path)
        for k in range(len(ag_list)-1):
        # for k in range(1):
            raw_video, sr = librosa.load(raw_path, sr=None)
            # if k in range(6):
            #     break
            # ag_video = get_ag_video(raw_video, k)
            ag_video = get_ag_video(raw_video, k+1)
            # save audio
            sampleRate = 48000  # hertz
            w_name1 = join(c_path, ag_list[k])
            w_name2 = raw_list[j][:-4] + '_' + ag_list[k] + '.wav'
            w_name = join(w_name1, w_name2)
            # print(w_name)
            obj = wave.open(w_name, 'w')
            obj.setnchannels(2)  # mono
            obj.setsampwidth(2)
            obj.setframerate(sampleRate)
            obj.writeframesraw(ag_video)
            obj.close()

    # print(v_path)
