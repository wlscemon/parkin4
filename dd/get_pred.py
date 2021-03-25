import numpy as np
import os
from xgboost import XGBClassifier

# Root = r'D:\Beginner\parkinson'
# modelpath = os.path.join(Root, 'bst_model.json')
modelpath = 'bst_model.json'
Max = np.array([2.75833294e+02,9.63737562e+01,4.98184567e+02,2.47379752e+01,
 2.29994457e+01, 2.81200609e+01, 4.74683544e+00, 7.35585417e+03,
 3.72082292e+03, 1.90551806e+00, 1.72423958e+03, 1.31807292e+03,
 1.18797326e+02, 2.69354143e+01, 2.41784680e+01, 4.37265057e+01,
 8.06842842e+00, 2.37617694e+01, 5.34844817e+01, 8.83926793e+00,
 1.72423958e+03, 1.31807292e+03])
Min = np.array([ 1.00510151e+02,  7.05171245e-01,  1.20977620e+02,  9.61035220e+00,
  1.12053671e+01,  1.78352932e+01,  1.10035211e-01,  1.17140625e+02,
  0.00000000e+00,  0.00000000e+00,  1.42520833e+02,  0.00000000e+00,
 -6.64712233e+01,  2.34389954e-01,  2.19753227e+00,  2.64550781e+00,
  5.78353470e-02,  3.03875884e+00,  0.00000000e+00,  8.89613200e-02,
  0.00000000e+00,  0.00000000e+00])
# Max = np.array(df.max())[0:22]
# Min = np.array(df.min())[0:22]
# print(Max)
# print(Min)
model = XGBClassifier()
model.load_model(modelpath)


def get_pred_real(feature=None):

    if feature is None:
        feature = np.array([[142.4661806,25.60091102,192.216992,13.70033097,15.48213288,23.48861259,2.604166667,222.7916667,134.164362,1.072303922,285.9732143,138.8563914,68.89864132,13.10333814,15.55946435,24.31523572,2.876740864,11.20985101,28.29754087,4.829439089,285.9732143,138.8563914]])
    else:
        feature = np.array([feature])
    feature = (feature-Min)/(Max-Min)*2-1
    pred_proba = model.predict_proba(feature)
    pred = model.predict(feature)
    print(pred)
    print(pred_proba[0][1])
    return pred_proba[0][1]

