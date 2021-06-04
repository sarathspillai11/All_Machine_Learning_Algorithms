import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
def con_dec_tree(X_train, X_test, y_train, y_test):
    # lowess will return our "smoothed" data with a y value for at every x-value
    # Computing regularization path using the LARS
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    return y_pred

if __name__ == '__main__':
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Categorical.from_codes(iris.target, iris.target_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    con_dec_tree(X_train, X_test, y_train, y_test )