import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

def extratree_classifier(X, y, estimator, criterion, max_features):
    # Building the model
    extra_tree_forest = ExtraTreesClassifier(n_estimators=estimator, criterion=criterion, max_features=max_features)
    # Training the model
    extra_tree_forest.fit(X, y)
    # Computing the importance of each feature
    feature_importance = extra_tree_forest.feature_importances_

    # Normalizing the individual importances
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in  extra_tree_forest.estimators_], axis=0)

    return feature_importance_normalized


if __name__ == '__main__':
    # Loading the data
    df = pd.read_csv(r'datasets\tennis.csv')
    n_estimators = 5
    criterion = 'entropy'
    max_features = 2
    # Seperating the dependent and independent variables
    y = df['Play Tennis']
    X = df.drop('Play Tennis', axis=1)
    extratree_classifier(X, y, n_estimators, criterion, max_features)

