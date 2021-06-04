from xgboost import XGBClassifier
from savemodel import saveas_sav
from sklearn.model_selection import cross_validate


def XgBoostPredictor(dataframe,x_train, y_train, x_test,ticketId,numClasses=2):
    print('inside xgboost module')


    classifier1 = XGBClassifier(n_estimators=50000, objective='multi:softprob',num_class = numClasses,
                                random_state=1234)  # default n_estimators=100 trees objective="binary:logistic" i.e for binary
    classifier1.fit(x_train, y_train)
    print('completed training')
    y_pred = classifier1.predict(x_test)
    dataframe['predicted'] = y_pred
    saveas_sav(classifier1, 'XgBoost_' + ticketId + '.sav')
    return dataframe