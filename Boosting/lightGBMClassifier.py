import lightgbm as lgb
from savemodel import saveas_sav
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

def lightGBMPredictor(dataframe,x_train, y_train, x_test,ticketId,leavesList,alphaList,minDataInLeafList,maxDepthList,boostingTypeList,learningRateList,numClasses=2):

    # Normal prediction without hypertuning
    # d_train = lgb.Dataset(x_train, label=y_train)
    #
    # params = {}
    # params['learning_rate'] = 0.003
    # params['boosting_type'] = 'dart'
    # params['num_class'] = int(numClasses)
    # params['objective'] = 'multiclass'
    # params['metric'] = 'multi_logloss'
    # params['sub_feature'] = 0.5
    # params['num_leaves'] = 100
    # params['min_data'] = 50
    # params['max_depth'] = 20
    #
    # clf = lgb.train(params, d_train, 1000)
    # # Prediction
    # y_pred = clf.predict(x_test)

    if(leavesList==[]):leavesList = [100]
    if (alphaList == []): alphaList = [0.02]
    if (minDataInLeafList == []): minDataInLeafList = [50]
    if (maxDepthList == []): maxDepthList = [20]
    if (boostingTypeList == []): boostingTypeList = ['dart']
    if (learningRateList == []): learningRateList = [0.003]


    param_grid = {
        'num_leaves': leavesList,
        'reg_alpha': alphaList,
        'min_data_in_leaf': minDataInLeafList,
        'max_depth': maxDepthList,
        'boosting_type': boostingTypeList,
        'learning_rate': learningRateList
    }

    # Splitting the dataset into the Training set and Test set

    gkf = KFold(n_splits=5, shuffle=True, random_state=42).split(X=x_train, y=y_train)
    lgb_estimator = lgb.LGBMClassifier()
    gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=gkf)
    # #gsearch = RandomizedSearchCV(lgb_estimator, param_grid, cv=gkf, verbose=1, n_jobs=-1, n_iter=10)
    # print('X train values in gsearch : ',set(x_train.values))
    print('y train values in gsearch : ',list(y_train))
    gsearch.fit(X=x_train, y=y_train)


    lgb_model = gsearch.best_estimator_

    y_pred = lgb_model.predict(x_test)



    y_pred = [np.argmax(line) for line in y_pred]
    print('type of lgbm output :',type(y_pred))
    print('actual pred :',len(y_pred))
    dataframe['predicted'] = y_pred

    saveas_sav(lgb_model, 'LightGBM_' + ticketId + '.sav')
    return dataframe