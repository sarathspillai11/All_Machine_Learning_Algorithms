
import pomegranate
import pandas as pd
import numpy as np

from Encoding import LabelEncoding
from Scaling import StandardScaling

# df = pd.read_excel(r'D:\Personal\SmartIT\ID3SampleTrain.xlsx')
# 
# df.replace(r'\s+', np.nan, regex=True)
# X_train = df.iloc[0:25].values
# y_train = df.iloc[0:25, -1].values
# X_test = df.iloc[25:59, :-1].values
# 
# print()
# pred_array = np.array(['None']*(X_test.shape)[0])
# X_test = np.append(X_test,pred_array.transpose())
# print(X_test)
# 
# labelencoder = LabelEncoder()
# for i in range(X_train.shape[1]):
#     X_train[:, i] = labelencoder.fit_transform(X_train[:, i].astype(str))
# 
# y_train = labelencoder.fit_transform(y_train.astype(str))
# 
# 
# for i in range(X_test.shape[1]):
#     X_test[:, i] = labelencoder.fit_transform(X_test[:, i].astype(str))
# 
# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc =StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# 
# 
# model = pomegranate.BayesianNetwork.from_samples(X_train, algorithm='exact')
# 
# print(X_test)
# 
# print(model.predict(X_test))

def BayesianNetworkPredictor(dataframe,x_train, y_train, x_test):
    """

    :rtype: A full dataframe with all the dependent variables and the output parameter
    """
    # x_train, y_train, x_test = LabelEncoding(x_train, y_train, x_test)
    # x_train, x_test = StandardScaling(x_train, x_test)
    model = pomegranate.BayesianNetwork.from_samples(x_train, algorithm='exact')
    out = model.predict(x_test)
    return out