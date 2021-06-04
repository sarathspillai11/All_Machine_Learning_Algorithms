import numpy as np
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')
from savemodel import saveas_sav

def least_angle_regression(dataframe,X_train,y_train, X_test,numNonZeroCoeffs,ticketId):
    # lowess will return our "smoothed" data with a y value for at every x-value
    #Computing regularization path using the LARS 
    regr = linear_model.Lars(n_nonzero_coefs=numNonZeroCoeffs)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    dataframe['predicted'] = y_pred
    saveas_sav(regr, 'lar_' + ticketId + '.sav')
    return(dataframe)
        
    

# if __name__ == '__main__':
#     from sklearn import datasets
#     diabetes = datasets.load_diabetes()
#     X = diabetes.data
#     y = diabetes.target
#     X = X[:, np.newaxis, 2]
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
