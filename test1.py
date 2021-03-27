import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
# import numpy as np
import pandas as pd
# import matlab.engine
# import io
import numpy as np
# import midi
#from scipy.io import wavfile
import scipy
#import matplotlib.pyplot as plt
from feature2pred import get_pred
from getfeature_py import getfeature
from predict import get_score
from get_pred import get_pred_real
import time

st.title(':musical_note: 帕金森检测')
# st.header('请选择要检测的声音文件，我们将预测您的健康状况')
st.subheader('请选择要检测的声音文件，我们将预测您的健康状况。')
# st.text('请选择要检测的声音文件，我们将预测您的健康状况')
# st.title(":musical_note: Convert a MIDI file to WAV")

uploaded_file = st.file_uploader("请上传 WAV 格式的音频文件", type=["wav"])
# st.write(type(uploaded_file))


if uploaded_file is None:
    # st.info("请上传 WAV 格式的音频文件")
    st.stop()

# dalay module
with st.spinner('Wait for it...'):
    samplerate, data = scipy.io.wavfile.read(uploaded_file)
    # st.write(uploaded_file)
    wav_name = 'audios/temp_audio.wav'
    scipy.io.wavfile.write(wav_name, samplerate, data)
    # st.write(data.shape[0])

    # length = data.shape[0] / samplerate

    # audio_test = 'D:\datas\parkinson\\1\\raw\\NC_001_1_2.wav'
    # Wave_write.writeframes(data)

    ## matlab part
    # eng = matlab.engine.start_matlab()
    # resultt = eng.get_feature(7.7)
    # feature_m = np.array([0,0,0,0])
    # feature_m = eng.getfeature_m(samplerate, data)
    # feature_m = eng.getfeature_m(wav_name)
    feature_p = getfeature(wav_name)
    # print(resultt)
    # eng.quit()
    # pred = get_pred(resultt)

    # time.sleep(1)
    st.success('Done!')
    # st.write(f"feature from matlab = {feature_m }")
    feature_p = np.array(feature_p)
    pred = get_pred_real(feature_p)
    # st.write(f"feature from python = {type(feature_p)}")
    # st.write(f"feature from python = {len(feature_p)}")
    # st.write(f"feature from python = {feature_p[0]}")
    # st.write(f"feature from python = {type(feature_p)}")
    # st.write(f"feature from matlab = {feature_m2}")
    # print(samplerate)
    # st.write([1,2,3,4])
    # st.write(f"data shape :{data.shape[0]}")
    # st.write(f"length = {length}s")
    # st.write(f"num of feature = {resultt}")
    # st.write(f"predict score = {pred}")

# progress show bar
# my_bar = st.progress(0)
# for percent_complete in range(100):
#     my_bar.progress(percent_complete + 1)

st.header(':musical_note: 您的预测结果为： ')
pred = int(pred*100)
st.title(pred)

if pred > 50:
    warn_text = '您有很大的可能患有帕金森综合征，建议及时至医院复查就诊。'
    st.subheader(warn_text)
    st.write('（分数越高，患病几率越大；分数越低，患病几率越小。）')
else:
    warn_text = '您的检测结果非常健康！'
    st.subheader(warn_text)
    st.write('（分数越高，患病几率越大；分数越低，患病几率越小。）')
    time.sleep(5)
    st.balloons()

st.header(':musical_note: 您的声音图像为： ')
chart_data = pd.DataFrame(
    np.random.randn(2009, 1),
    columns=['voice'])

st.line_chart(chart_data)

# use matplotlib
# fig, ax = plt.subplots()
# ax.scatter([1, 2, 3], [1, 2, 3])
#
# st.pyplot(fig)


st.header(':musical_note: 您的声音信号指标为： ')
feature_list = ['mF0', 'sF0', 'mmF0', 'mlogE', 'slogE', 'mmlogE', 'Vrate', 'avgdurv', 'stddurv', 'Silrate',
                'avgdurs', 'stddurs', 'F0varsemi', 'mjitter', 'mshimmer', 'apq', 'ppq',
                'mlogE', 'degreeU', 'varF0', 'avgdurs', 'stddurs']
df1 = pd.DataFrame(
    # np.random.randn(1, 6),
    feature_p[:6].reshape(1, -1),
    # columns=('col %d' % i for i in range(20)))
    columns=feature_list[:6])

df2 = pd.DataFrame(
    # np.random.randn(6, 13),
    feature_p[6:12].reshape(1, -1),
    # columns=('col %d' % i for i in range(20)))
    columns=feature_list[6:12])
df3 = pd.DataFrame(
    # np.random.randn(1, 6),
    feature_p[13:19].reshape(1, -1),
    # columns=('col %d' % i for i in range(20)))
    columns=feature_list[13:19])
df4 = pd.DataFrame(
    # np.random.randn(1, 6),
    feature_p[19:].reshape(1, -1),
    # columns=('col %d' % i for i in range(20)))
    columns=feature_list[19:])
st.dataframe(df1)  # Same as st.write(df)
st.dataframe(df2)  # Same as st.write(df)
st.dataframe(df3)  # Same as st.write(df)
st.dataframe(df4)  # Same as st.write(df)






# ui = wavfile.read(uploaded_file, mmap=False)
# st.write(type(ui))
# st.write(uploaded_file.getparams())
# midi_data = midi.PrettyMIDI(uploaded_file)
# audio_data = midi_data.fluidsynth()
# audio_data = np.int16(audio_data / np.max(np.abs(
#     audio_data)) * 32767 * 0.9)  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py
#
# virtualfile = io.BytesIO()
# wavfile.write(virtualfile, 44100, audio_data)
#
# st.audio(virtualfile)
# st.markdown("Download the audio by right-clicking on the media player")


# print('dada')


# bonus

st.write('您的检测结果非常健康！')
time.sleep(8)
st.balloons()
