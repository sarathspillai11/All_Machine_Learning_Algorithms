# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 02:07:54 2020

@author: vamsi
"""

#https://medium.com/analytics-army/deep-belief-networks-an-introduction-1d52bb867a25
#https://github.com/albertbup/deep-belief-network

from dbn.tensorflow import SupervisedDBNClassification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

def radialbasis(x_train,y_train,x_test,hidden_layers_structure = [256, 256], learning_rate_rbm=0.05, learning_rate=0.1,
    n_epochs_rbm=10,
    n_iter_backprop=100,
    batch_size=32,
    activation_function='relu',
    dropout_p=0.2):
    # fitting RBF-Network with data
    classifier = SupervisedDBNClassification(hidden_layers_structure, learning_rate_rbm, learning_rate,
    n_epochs_rbm,
    n_iter_backprop,
    batch_size,
    activation_function,
    dropout_p)
    classifier.fit(x_train, y_train)
    y_hat = classifier.predict(x_test)
    return y_hat