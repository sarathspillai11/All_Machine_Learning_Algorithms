# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 01:08:51 2020

@author: vamsi
"""
# referemce from here

import numpy as np
import matplotlib.pyplot as plt
from RBFN import RBFN

def radialbasis(x_train,y_train,x_test,hidden_shape=10, sigma=1.):
    # fitting RBF-Network with data
    model = RBFN(hidden_shape, sigma)
    model.fit(x_train,y_train)
    # Print the model
    print(model.trace())
    print(model.summary())

    # make prediction
    y_hat = model.predict(x_test)

    return y_hat