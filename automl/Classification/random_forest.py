from sklearn.ensemble import RandomForestClassifier
import pandas as pd

configdf = pd.read_excel('config.xlsx', sheet_name='CONFIG')
def random_forest(x_train, y_train, x_test):
    iter = configdf['var'][7].astype(int)
    Method = configdf['Method'][7]
    clasi = RandomForestClassifier(n_estimators=iter)
    clasi.fit(x_train, y_train)
    out = clasi.predict(x_test)
    return out