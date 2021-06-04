from sklearn.tree import DecisionTreeClassifier
import pandas as pd

configdf = pd.read_excel('config.xlsx', sheet_name='CONFIG')
def decision_tree(x_train, y_train, x_test):
    iter = configdf['var'][6].astype(int)
    Method = configdf['Method'][6]
    clasi = DecisionTreeClassifier(criterion=Method, max_depth=iter)
    clasi.fit(x_train, y_train)
    out = clasi.predict(x_test)
    return out