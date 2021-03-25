from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import random
import math
import sys, os
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

Root = r'D:\Beginner\parkinson'
csvpath = os.path.join(Root, 'data01.csv')
if __name__ == '__main__':
    # df = pd.read_csv(csvpath,index_col=0)
    # features = df.loc[:, df.columns != 'label'].values[:, 1:]
    # labels = df.loc[:, 'label'].values
    # scaler = MinMaxScaler((-1, 1))
    # X = scaler.fit_transform(features)
    # y = labels
    # 数据处理方式略有不同
    df = pd.read_csv(csvpath, header=None, index_col=0).values
    labels = df[1:, 22]
    y = labels
    X = df[1:, 1:22]
    X = pd.DataFrame(X)
    n_samples, n_features = X.shape
    X = pd.DataFrame(X).values
    X = scale(X)

    ratio_c1 = [0 for i in range(0, 21)]
    ratio_c2 = [0 for i in range(0, 21)]
    print(X)


    def dist(x, y):
        dis = 0.0
        for i in range(0, 21):
            dis += (x[i] - y[i]) * (x[i] - y[i])
        return dis


    def KmeansClustering():
        best_acc = 0.0
        c1 = [0 for i in range(0, 21)]
        c2 = [0 for i in range(0, 21)]
        best_c1 = [0 for i in range(0, 21)]
        best_c2 = [0 for i in range(0, 21)]
        best_c1_data_points = []
        best_c2_data_points = []
        t = 10
        while t > 0:
            c1 = X[random.randint(0, 251)]
            c2 = X[random.randint(0, 251)]
            # print c1
            # print c2
            c1data = [0 for i in range(0, 21)]
            c2data = [0 for i in range(0, 21)]
            c1_count = 0
            c2_count = 0
            c1_data_points = []
            c2_data_points = []
            for j in range(0, len(X)):
                sample = X[j]
                d1 = dist(sample, c1)
                d2 = dist(sample, c2)
                if d1 < d2:
                    c1_count += 1
                    for i in range(0, 21):
                        c1data[i] += sample[i]
                    c1_data_points.append(j)
                else:
                    c2_count += 1
                    for i in range(0, 21):
                        c2data[i] += sample[i]
                    c2_data_points.append(j)
                    # print c1_count
            # print c2_count
            for i in range(0, 21):
                c1[i] = (c1data[i] * 1.0) / c1_count
            for i in range(0, 21):
                c2[i] = (c2data[i] * 1.0) / c2_count
            # print c1
            # print c2
            # print t
            # assume c1 to be 1 cluster
            correct = 0
            total = 252
            i = 0
            for sample in X:
                d1 = dist(sample, c1)
                d2 = dist(sample, c2)
                # print d1
                # print d2

                if d1 < d2:
                    if labels[i] == "1":
                        correct += 1
                else:
                    if labels[i] == "0":
                        correct += 1
                i += 1
            # print correct
            acc = correct / 252.0
            if acc > best_acc:
                best_acc = acc
                best_c1_data_points = c1_data_points[:]
                best_c2_data_points = c2_data_points[:]
                for j in range(0, 21):
                    best_c1[j] = c1[j]
                    best_c2[j] = c2[j]
            t = t - 1
        best_c1 = np.array(best_c1)
        best_c2 = np.array(best_c2)

        means_1 = []
        means_2 = []
        for i in range(0, 21):
            means_1.append(0)
            means_2.append(0)

        for datapoint in best_c1_data_points:
            for j in range(0, 21):
                means_1[j] += X[datapoint][j]

        for datapoint in best_c2_data_points:
            for j in range(0, 21):
                means_2[j] += X[datapoint][j]

        for j in range(0, 21):
            means_1[j] = (1.0 * means_1[j]) / len(best_c1_data_points)

        for j in range(0, 21):
            means_2[j] = (1.0 * means_2[j]) / len(best_c2_data_points)

        for i in range(0, 21):
            ratio_c1[i] = (means_1[i] * 1.0) / best_c1[i]
            ratio_c2[i] = (means_2[i] * 1.0) / best_c2[i]
        # np.array(ratio_c1)
        # np.array(ratio_c2)

        c1_rat = np.array(ratio_c1)
        c2_rat = np.array(ratio_c2)
        # print c1_rat
        # print c2_rat
        i = 0
        for x in range(0, 252):
            d1 = dist(sample, c1)
            d2 = dist(sample, c2)
            if d1 < d2:
                if labels[i] == "1":
                    for i in range(0, 21):
                        X[x][i] *= c1_rat[i]
            else:
                if labels[i] == "0":
                    for i in range(0, 21):
                        X[x][i] *= c2_rat[i]
            i += 1


    KmeansClustering()  # data转换

    test_times = 10
    avr_res = 0
    avr_accu = 0.0
    start = time.perf_counter()
    selfincrese = 0
    while (test_times):
        count = 0
        while 1:
            x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.05, random_state=selfincrese)
            count += 1
            selfincrese += 1
            # model = tree.DecisionTreeClassifier()
            # model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=None)
            model = KNeighborsClassifier()
            # model = LogisticRegression(random_state=0, solver=''newton-cg', multi_class='multinomial')
            model.fit(x_train, y_train)

            y_prediction = model.predict(x_test)
            avr_accu += accuracy_score(y_test, y_prediction) * 100
            # print("Accuracy Score is", accuracy_score(y_test, y_prediction) * 100)
            if accuracy_score(y_test, y_prediction) * 100 > 90:
                # print(count)
                break
        test_times -= 1
        avr_res += count
    avr_accu /= avr_res
    avr_res /= 10
    end = time.perf_counter()
    # print(csvpath)
    print('Average tests: %s Tests' % avr_res)
    print('Average accuracy:%.2f%%' % avr_accu)
    print('Running time: %f Seconds' % (end - start))
    print(datetime.datetime.now())