from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from savemodel import saveas_sav
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def random_forest(dataframe,x_train, y_train, x_test,ticketId,estimatorsList,maxFeaturesList,maxDepthList,minSamplesSplitList,minSamplesLeafList,bootStrapList,numClasses=2):
    #configdf = pd.read_excel(r'D:\Personal\DataScience\Configurations\config.xlsx', sheet_name='CONFIG')
    #iter = configdf['var'][7].astype(int)
    #Method = configdf['Method'][7]

    ########## used before
    # clasi = RandomForestClassifier(n_estimators=1000)
    # clasi.fit(x_train, y_train)
    # out = clasi.predict(x_test)
    # hyper parameter tuning for best parameters

    if (estimatorsList == []): estimatorsList = [500]
    if (maxFeaturesList == []): maxFeaturesList = ['auto']
    if (minSamplesSplitList == []): minSamplesSplitList = [2]
    if (maxDepthList == []): maxDepthList = [10]
    if (minSamplesLeafList == []): minSamplesLeafList = [2]
    if (bootStrapList == []): bootStrapList = [True]

    # Create the random grid
    random_grid = {'n_estimators': estimatorsList,
                   'max_features': maxFeaturesList,
                   'max_depth': maxDepthList,
                   'min_samples_split': minSamplesSplitList,
                   'min_samples_leaf': minSamplesLeafList,
                   'bootstrap': bootStrapList}
    print(random_grid)


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=10, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit on training data
    rf_random.fit(x_train, y_train)

    classifier = rf_random.best_estimator_

    out = classifier.predict(x_test)

    dataframe['predicted'] = out
    saveas_sav(classifier, 'randomForest_' + ticketId + '.sav')
    return dataframe