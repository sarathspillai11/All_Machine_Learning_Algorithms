# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:14:17 2020

@author: vamsi
"""


from category_encoders import *
import pandas as pd
from sklearn.datasets import load_boston


def cat_encoder(xtrain,colnames):
    # use target encoding to encode two categorical features
    enc = TargetEncoder(cols=colnames)
    
    xtrain = enc.transform(xtrain)
    return xtrain
if __name__ == '__main__':
    train = pd.read_csv(r'C:\machinlearning-trainings-master\allregressions\decisiontree.csv')
    colnames = ['CHAS', 'RAD']
    d = cat_encoder(train,colnames)
