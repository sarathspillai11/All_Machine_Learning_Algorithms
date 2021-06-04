import numpy as np
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')

def least_angle_regression(X_train, X_test, y_train, y_test):
    # lowess will return our "smoothed" data with a y value for at every x-value
    #Computing regularization path using the LARS 
    regr = linear_model.Lars(n_nonzero_coefs=1)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    return(y_pred)
        
    

if __name__ == '__main__':
    from sklearn import datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X = X[:, np.newaxis, 2]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
