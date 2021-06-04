from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from savemodel import saveas_sav

#X_train, y_train, x_test, optimization, CrossValidation , cv=3, n_jobs=-1
def logistic_regression(dataframe,X_train, y_train, x_test, optimization=None, CrossValidation=None , cv=None, n_jobs=None,max_iter=[100],ticketId=''):
    configdf = pd.read_excel(r'D:\Personal\DataScience\Configurations\config.xlsx', sheet_name='CONFIG')
    iter = configdf['var'][1].astype(int)
    regressor = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, max_iter=iter, multi_class='ovr', n_jobs=1,
                                   penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
                                   verbose=0, warm_start=False)
    if optimization:
        kfold = KFold(n_splits=3, random_state=7)
        result = cross_val_score(regressor,X_train, y_train, cv=kfold, scoring='accuracy')
        dual = [True, False]
        max_iter = [100, 110, 120, 130, 140]
        param_grid = dict(dual=dual, max_iter=max_iter)
        if CrossValidation == 'Grid':

            # l2 optimization
            lr = LogisticRegression(penalty='l2')
            regressor = GridSearchCV(estimator=lr, param_grid=param_grid, cv=cv, n_jobs=n_jobs)
            # start_time = time.time()
            result = regressor.fit(X_train, y_train)
            # Summarize results
            # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            # print("Execution time: " + str((time.time() - start_time)) + ' ms')
        if CrossValidation == "Random":
            random = RandomizedSearchCV(estimator=lr, param_distributions=param_grid, cv=cv, n_jobs=n_jobs)
            # start_time = time.time()
            result = random.fit(X_train, y_train)
            # Summarize results
            # print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
            # print("Execution time: " + str((time.time() - start_time)) + ' ms')

    else:
        regressor.fit(X_train, y_train)

    y_pred = regressor.predict(x_test)
    dataframe['predicted'] = y_pred
    saveas_sav(regressor, 'logisticRegression_' + ticketId + '.sav')
    return dataframe