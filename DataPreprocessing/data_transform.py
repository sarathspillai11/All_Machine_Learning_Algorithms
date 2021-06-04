# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:31:11 2020

@author: vamsi
"""


import pandas as pd
import numpy as np
def data_transform(train, method):
    train = train.transform(method)
    return train


if __name__ == '__main__':
    train = pd.read_csv(r'C:\machinlearning-trainings-master\allregressions\decisiontree.csv')
    method = [np.exp]
    d = data_transform(train,method)
    print(d)