from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from savemodel import saveas_sav

def decision_tree(dataframe,x_train, y_train, x_test,ticketId):
    configdf = pd.read_excel(r'D:\Personal\DataScience\Configurations\config.xlsx', sheet_name='CONFIG')
    iter = configdf['var'][6].astype(int)
    Method = configdf['Method'][6]
    clasi = DecisionTreeClassifier(criterion=Method, max_depth=iter)
    clasi.fit(x_train, y_train)
    out = clasi.predict(x_test)
    dataframe['predicted'] = out
    saveas_sav(clasi, 'decision_tree_' + ticketId + '.sav')
    return dataframe