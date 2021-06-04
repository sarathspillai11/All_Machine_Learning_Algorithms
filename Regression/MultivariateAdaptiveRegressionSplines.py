import numpy
from pyearth import Earth
from matplotlib import pyplot
from savemodel import saveas_sav
from automl.Optimization.adjusted_r_square import backwardEliminationwithadjustedrsquare
from sklearn.model_selection import train_test_split

def MarsImplementation(dataframe,x_train,y_train,x_test,SL=None, size=None, randomstate=None, optimization=False,ticketId=''):
    # Fit an Earth model
    model = Earth()
    if optimization:
        X_opt = x_train[:, list(range(x_train.shape[1] - 1))]
        X_Modeledwithadjustedrsquare = backwardEliminationwithadjustedrsquare(X_opt,y_train, SL)
        X_linopt_train, X_linopt_test, y_linopt_train, y_linopt_test = train_test_split(X_Modeledwithadjustedrsquare, y_train,
                                                                                        test_size=size,
                                                                                        random_state=randomstate)
        model.fit(X_linopt_train, y_linopt_train)
    else:

        model.fit(x_train, y_train)
    print('inside mars func')
    # Print the model
    print(model.trace())
    print(model.summary())

    # make prediction
    y_hat = model.predict(x_test)
    dataframe['predicted'] = y_hat
    saveas_sav(model, 'mars_' + ticketId + '.sav')
    return dataframe