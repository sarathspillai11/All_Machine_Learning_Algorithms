# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:15:00 2020

@author: vamsi
"""

import pandas as pd
from sklearn.linear_model import Lasso
from savemodel import saveas_sav
def lasso_(dataframe,x_train, y_train, x_test,ticketId):
    print('in lasso module')
    configdf = pd.read_excel(r'D:\Personal\DataScience\Configurations\config.xlsx', sheet_name='CONFIG')
    alfa = configdf['var'][3].astype(int)
    regressor = Lasso(alpha=alfa)
    regressor.fit(x_train, y_train)
    out = regressor.predict(x_test)
    dataframe['predicted'] = out
    saveas_sav(regressor, 'lasso_' + ticketId + '.sav')
    return dataframe



