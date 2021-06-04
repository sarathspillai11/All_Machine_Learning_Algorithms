from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from savemodel import saveas_sav

def ridgeRegression(dataframe,x_train, y_train, x_test,ticketId):
    print('in ridge func module')
    regressor = Ridge()
    parameters = {'alpha': [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2, 1, 5, 10, 20]}
    ridge_regressor = GridSearchCV(regressor, parameters, scoring="neg_mean_squared_error", cv = 5)
    ridge_regressor.fit(x_train, y_train)
    out = ridge_regressor.predict(x_test)
    dataframe['predicted'] = out
    saveas_sav(ridge_regressor, 'Ridge_' + ticketId + '.sav')
    return dataframe

