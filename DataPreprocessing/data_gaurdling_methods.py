# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:26:06 2020

@author: vamsi
"""


def drop_highcardinality(train):
    tempdict = {}
    for col in train:
        cardinality = len(pd.Index(train[col]).value_counts())
        tempdict[train[col].name] = cardinality
    #print(max(dict.values()))
    listOfKeys = [key  for (key, value) in tempdict.items() if value == max(tempdict.values())]
    train.drop(listOfKeys, axis=1)
    return train
        


if __name__ == '__main__':
    import pandas as pd
    train = pd.read_csv(r'C:\machinlearning-trainings-master\allregressions\decisiontree.csv')
    d = drop_highcardinality(train)
    


