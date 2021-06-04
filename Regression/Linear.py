from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from automl.Optimization.adjusted_r_square import backwardEliminationwithadjustedrsquare
from savemodel import saveas_sav
#x_train, y_train, x_test, SL =0.5, size = 0.33, randomstate = 33, optimization = True"
def linearRegressor(dataframe,x_train, y_train, x_test, SL=None, size=None, randomstate=None, optimization=False,ticketId=''):
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    print('inside linear regressor')

    regressor = LinearRegression()
    if optimization:
        X_opt = x_train[:, list(range(x_train.shape[1] - 1))]
        X_Modeledwithadjustedrsquare = backwardEliminationwithadjustedrsquare(X_opt,y_train, SL)
        X_linopt_train, X_linopt_test, y_linopt_train, y_linopt_test = train_test_split(X_Modeledwithadjustedrsquare, y_train,
                                                                                        test_size=size,
                                                                                        random_state=randomstate)
        regressor.fit(X_linopt_train, y_linopt_train)
    else:
        regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    dataframe['predicted'] = y_pred
    saveas_sav(regressor, 'linearRegression_' + ticketId + '.sav')
    return dataframe



