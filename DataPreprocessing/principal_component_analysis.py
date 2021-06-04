# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:41:26 2020

@author: vamsi
"""

from sklearn.decomposition import PCA
def principalcomponentanalysis(xtrain,no_of_components):
    pca_ = PCA(n_components=no_of_components)
    xtrain = pca_.fit_transform(xtrain)
    return xtrain

if __name__ == '__main__':
    import pandas as pd
    train = pd.read_csv(r'C:\machinlearning-trainings-master\allregressions\decisiontree.csv')
    no_of_components = 2
    d = principalcomponentanalysis(train,no_of_components)
    