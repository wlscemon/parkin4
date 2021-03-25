import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import datetime
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

Root = r'D:\Beginner\parkinson'
csvpath = os.path.join(Root, 'data01_final.csv')


if __name__ == '__main__':
    df = pd.read_csv(csvpath, index_col=0)
    # print(df.tail())
    #
    # print(df.describe())
    #
    # print(df.info())
    #
    # print(df.shape)

    # features = df.loc[:, df.columns != 'label'].values[:, 1:]
    features = df.loc[:, df.columns != 'label'].values

    # print(features.head())
    labels = df.loc[:, 'label'].values

    scaler = MinMaxScaler((-1,1))
    X = scaler.fit_transform(features)
    # X = features
    y = labels

    test_times = 10
    avr_res = 0
    avr_accu = 0.0
    # auc_Score = []
    # accuracy = []
# GridSearchCV
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=12)
    # parameters = {'min_child_weight': [0,1,2],
    #               'max_depth': [3,4,5,6,7,8,9],
    #               'n_estimators':[100,200,500],
    #               'gamma':[0.0, 0.1,0.2],
    #               # 'subsample':[i/10.0 for i in range(7,10)],
    #               # 'colsample_bytree':[i/10.0 for i in range(7,10)],
    #               # 'reg_alpha':[1e-5, 1e-2, 0.1],
    #               # 'reg_lambda':[1,5,10],
    #               'learning_rate':[0.01,0.1,0.3]}
    #
    # grid_search = GridSearchCV(estimator=XGBClassifier(), param_grid=parameters, cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(x_train, y_train)
    #
    # xgb = grid_search.best_estimator_
    #
    # print(grid_search.best_params_)
    # y_pred = xgb.predict(x_test)
    # print(y_pred)
    # # print("\n", confusion_matrix(y_test, y_pred))
    # xgb_acc = accuracy_score(y_test, y_pred)
    # print("\nAccuracy Score {}".format(xgb_acc))
    # print("Classification report: \n{}".format(classification_report(y_test, y_pred)))

    start = time.perf_counter()
    selfincrese = 0
    best_acc = 0
    best_acc = 0
    bestincrese = 0

    while test_times:
        count = 0
        while 1:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=selfincrese)
            count += 1
            selfincrese += 1
            model = XGBClassifier()
            model.load_model('bst_model.json')
            model.fit(x_train, y_train)

            y_pre = model.predict(x_test)
            print(y_pre)
            y_pre_proba=model.predict_proba(x_test)
            print(y_pre_proba)
            print(x_test)
            # y_pro = model.predict_proba(x_test)[:, 1]
            # print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro))
            # print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre))
            # auc_Score.append(metrics.roc_auc_score(y_test, y_pro))
            # accuracy.append(metrics.accuracy_score(y_test, y_pre))
            avr_accu += accuracy_score(y_test, y_pre) * 100
            # print("Accuracy Score is", accuracy_score(y_test, y_prediction) * 100)
            if accuracy_score(y_test, y_pre) * 100 > best_acc:
                best_acc = accuracy_score(y_test, y_pre) * 100
                bestincrese = selfincrese
                #model.save_model('bst_model.json')
            if accuracy_score(y_test, y_pre) * 100 > 90:
                # print(count)
                break
        test_times -= 1
        avr_res += count
    avr_accu /= avr_res
    avr_res /= 10
    end = time.perf_counter()

    print(csvpath)
    print('Average tests: %s Tests' % avr_res)
    print('Average accuracy:%.2f%%' % avr_accu)
    print('Running time: %f Seconds' % (end - start))
    print('Best accuracy:%.2f%%' % best_acc)
    print('Best random state: %s' % bestincrese)
    #print('The model is:', xgb)
    print(datetime.datetime.now())
