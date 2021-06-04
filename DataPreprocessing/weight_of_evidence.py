# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 00:26:25 2020

@author: vamsi
"""


import warnings
warnings.filterwarnings('ignore')
class switch(object):
    value = None
    def __new__(class_, value):
        class_.value = value
        return True

def case(*args):
    return any((arg == switch.value for arg in args))

def w_o_e(pswitch, X,y):
    while switch(pswitch):
        if case('MonotonicBinning'):
            from xverse.transformer import MonotonicBinning
            clf = MonotonicBinning()
            clf.fit(X, y)
            print(clf.bins)
            output_bins = clf.bins 
            return output_bins
            break
        if case('Weight_of_Evidence'):
            from xverse.transformer import WOE
            clf = WOE()
            clf.fit(X, y)
            clf.woe_df
            #information value dataset
            clf.iv_df
            clf.transform(X)
            return clf
            break
        if case('VotingSelector'):
          clf = VotingSelector()
          clf.fit(X, y)
          print(clf.feature_importances_)
          print(clf.feature_votes_)
          print(clf.transform(X).head())
          return clf
          break

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Categorical.from_codes(iris.target, iris.target_names)
    d = w_o_e('MonotonicBinning',X,y)
    print(d)