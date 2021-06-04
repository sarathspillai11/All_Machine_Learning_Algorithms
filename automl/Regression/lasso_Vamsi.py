# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:15:00 2020

@author: vamsi
"""

import pandas as pd
from sklearn.linear_model import Lasso
configdf = pd.read_excel('config.xlsx', sheet_name='CONFIG')
def lasso_(x_train, y_train, x_test):
    alfa = configdf['var'][3].astype(int)
    regressor = Lasso(alpha=alfa)
    regressor.fit(x_train, y_train)
    out = regressor.predict(x_test)
    return out



