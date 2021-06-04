# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:27:12 2020

@author: vamsi
"""


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from savemodel import saveas_sav

def polynomial_feature_regression(dataframe,x_train, y_train, x_test,polynomialDegree,ticketId):
    print('in poly module')
    # configdf = pd.read_excel(r'D:\Personal\DataScience\Configurations\config.xlsx', sheet_name='CONFIG')
    # dgree = configdf['var'][2].astype(int)
    poly_reg = PolynomialFeatures(degree=polynomialDegree)
    X_poly = poly_reg.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(X_poly,y_train)
    out = regressor.predict(x_test)
    dataframe['predicted'] = out
    saveas_sav(regressor, 'poly_regression_' + ticketId + '.sav')
    return dataframe





