# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:04:54 2020

@author: vamsi
"""
from sklearn.decomposition import TruncatedSVD
def principalcomponentanalysis(xtrain, no_of_components, iter, random):
    svd = TruncatedSVD(n_components=no_of_components, n_iter=iter, random_state=random)
    svd.fit(xtrain)
    return xtrain

if __name__ == '__main__':
    import pandas as pd
    train = pd.read_csv(r'C:\machinlearning-trainings-master\allregressions\decisiontree.csv')
    no_of_components = 2
    n_iter=7
    random_state=42
    d = principalcomponentanalysis(train,no_of_components,n_iter, random_state)
    