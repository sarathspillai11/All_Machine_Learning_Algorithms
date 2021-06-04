import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as numpy
from savemodel import saveas_sav


def RBFNClassifier(dataframe,x_train, y_train, x_test,y_test,numClusters=8,ticketId=''):
    print('inside classifier')
    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train = scaler.transform(x_train)
    X_test = scaler.transform(x_test)
    print('before clustering')
    K_cent = numClusters
    km = KMeans(n_clusters=K_cent, max_iter=100)
    km.fit(X_train)
    cent = km.cluster_centers_
    print('after clustering')
    max = 0
    for i in range(K_cent):
        for j in range(K_cent):
            d = numpy.linalg.norm(cent[i] - cent[j])
            if (d > max):
                max = d
    d = max

    sigma = d / math.sqrt(2 * K_cent)

    shape = X_train.shape
    row = shape[0]
    column = K_cent
    G = numpy.empty((row, column), dtype=float)
    for i in range(row):
        for j in range(column):
            dist = numpy.linalg.norm(X_train[i] - cent[j])
            G[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * sigma, 2))

    GTG = numpy.dot(G.T, G)
    GTG_inv = numpy.linalg.inv(GTG)
    fac = numpy.dot(GTG_inv, G.T)
    W = numpy.dot(fac, y_train)

    row = X_test.shape[0]
    column = K_cent
    G_test = numpy.empty((row, column), dtype=float)
    for i in range(row):
        for j in range(column):
            dist = numpy.linalg.norm(X_test[i] - cent[j])
            G_test[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * sigma, 2))

    prediction = numpy.dot(G_test, W)
    prediction = 0.5 * (numpy.sign(prediction - 0.5) + 1)

    score = accuracy_score(prediction, y_test)
    print(score.mean())

    dataframe['predicted'] = list(prediction)

    return dataframe

