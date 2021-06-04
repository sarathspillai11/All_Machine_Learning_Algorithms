import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from savemodel import saveas_sav
def con_dec_tree(dataframe,X_train,y_train, X_test,ticketId):
    # lowess will return our "smoothed" data with a y value for at every x-value
    # Computing regularization path using the LARS
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    dataframe['predicted'] = y_pred
    saveas_sav(dt, 'conditional_decision_tree_' + ticketId + '.sav')
    return dataframe

# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.datasets import load_iris
#     from sklearn.model_selection import train_test_split
#     iris = load_iris()
#     X = pd.DataFrame(iris.data, columns=iris.feature_names)
#     y = pd.Categorical.from_codes(iris.target, iris.target_names)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#     con_dec_tree(X_train, X_test, y_train, y_test )