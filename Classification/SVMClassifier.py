

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Fitting SVM to the Training set
from sklearn.svm import SVC
from savemodel import saveas_sav
def svmClassification(dataframe,x_train, y_train, x_test,ticketId,cList,gammaList,kernelList):
    if(len(cList) == 0):
        cList = [0.5]
    if (len(gammaList) == 0):
        gammaList = [0.1]
    if (len(kernelList) == 0):
        kernelList = ['rbf']
    #param_grid = {'C': [0.1, 0.5, 1, 1.5, 10], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    param_grid = {'C': cList, 'gamma': gammaList, 'kernel': kernelList}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid.fit(x_train, y_train)

    print('best parameters :')
    print(grid.best_params_)

    print('how model looks after tuning :')
    print(grid.best_estimator_)

    # classifier = LinearSVC(C=0.5, class_weight=None, dual=True, fit_intercept=True,
    #                        intercept_scaling=1, loss='squared_hinge', max_iter=10000,
    #                        multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
    #                        verbose=0)



    classifier = grid.best_estimator_
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    dataframe['predicted'] = y_pred
    saveas_sav(classifier, 'svm_' + ticketId + '.sav')

    return dataframe


# if __name__ == '__main__':
#     data = pd.read_csv(r"D:\Personal\SmartIT\data\BankNote_Authentication.csv")
#     # test = pd.read_excel(r"D:\Personal\SmartIT\data\hematological malignancies bayesian.xls",sheet_name='BCCA')
#
#     trainingColumns = (list(data.columns))[:-2]
#     print('training col :', trainingColumns)
#     outputColumn = (list(data.columns))[-1]
#     print('output column :', outputColumn)
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#
#     dataframe = svmClassification(X_test, X_train, y_train, X_test, '123', cList = [] , gammaList = [], kernelList= [])
#
#     from sklearn.metrics import confusion_matrix
#     from sklearn.metrics import accuracy_score
#     from sklearn.metrics import classification_report
#
#     y_pred = dataframe['predicted']
#
#     cm = confusion_matrix(y_test, y_pred)
#     # Accuracy
#
#     accuracy = accuracy_score(y_test, y_pred)
#
#     print(cm)
#
#     print(accuracy)
#
#     print(classification_report(y_test, y_pred))




