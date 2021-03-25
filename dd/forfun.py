import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
# import numpy as np
# import pandas as pd
import matlab.engine
# import io
import matplotlib.pyplot as plt
import numpy as np
# from bokeh.plotting import figure
# import midi
from scipy.io import wavfile
from feature2pred import get_pred
from predict import get_score
from time import time


st.title('My first adsdspp')
st.text('This will appear first')
# Appends some text to the app.

my_slot1 = st.empty()
# Appends an empty slot to the app. We'll use this later.

my_slot2 = st.empty()
# Appends another empty slot.

st.text('This will appear last')
# Appends some more text to the app.

my_slot1.text('This will appear second')
# Replaces the first empty slot with a text string.

my_slot2.line_chart(np.random.randn(20, 2))
# Replaces the second empty slot with a chart.






fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [1, 2, 3])

st.pyplot(fig)