# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:15:06 2020

@author: vamsi
"""

import pandas as pd
import numpy as np
def statistic_feature_identification(train):
    profile = train.describe()
    variance = np.var(train)
    skewness = train.skew(axis = 0, skipna = True)
    kurtosis = train.kurtosis(axis = 0, skipna = True)
    return profile, variance, skewness, kurtosis


if __name__ == '__main__':
    train = pd.read_csv(r'C:\machinlearning-trainings-master\allregressions\decisiontree.csv')
    d = statistic_feature_identification(train)
    print(d)
    
