import pickle
import pandas as pd
from Encoding import LabelEncoding
from DataPreprocessing.scaler import ScaleTransform

data = pd.read_excel(r"D:\Personal\SmartIT\data\Credit Card Fraud\validate.xlsx")
trainingColumns = (list(data.columns))[:-1]
print('training col :', trainingColumns)
outputColumn = (list(data.columns))[-1]
print('output column :', outputColumn)
X1 = data.values
X = data.iloc[:, :-1]
#y = data.iloc[:, -1]
X = X.values

X = LabelEncoding.LabelEncode(X)

X = ScaleTransform(X,'standard')
Xg_Model = r'D:\Personal\SmartIT\DataScience_Latest\XgBoost_23082020_XgBoost.sav'
Rf_Model = r'D:\Personal\SmartIT\DataScience_Latest\randomForest_28082020_RandomForest.sav'
Lgbm_Model = r'D:\Personal\SmartIT\DataScience_Latest\LightGBM_23082020_LGBM.sav'
kmeans_Model = r'D:\Personal\SmartIT\DataScience_Latest\kmeans_23082020.sav'
dbscan_Model = r'D:\Personal\SmartIT\DataScience_Latest\dbScan_123.sav'
ann_Model = r'D:\Personal\SmartIT\DataScience_Latest\ann_23082020.sav'
'''
Xg_loaded_model = pickle.load(open(Xg_Model, 'rb'))

pred = Xg_loaded_model.predict(X)
XgBoost_out = data
XgBoost_out['predicted'] = list(pred)
XgBoost_out.to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\actual\final_XgBoost_23082020.xlsx')


Rf_loaded_model = pickle.load(open(Rf_Model, 'rb'))

pred = Rf_loaded_model.predict(X)
Rf_out = data
Rf_out['predicted'] = list(pred)
Rf_out.to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\actual\final_randomForest_23082020.xlsx')


lgbm_loaded_model = pickle.load(open(Lgbm_Model, 'rb'))

pred = lgbm_loaded_model.predict(X)
lgbm_out = data
lgbm_out['predicted'] = list(pred)
lgbm_out.to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\actual\final_lgbm_23082020.xlsx')
kmeans_loaded_model = pickle.load(open(kmeans_Model, 'rb'))

pred = kmeans_loaded_model.predict(X1)
kmeans_out = data
kmeans_out['predicted'] = list(pred)
kmeans_out.to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\actual\final_kmeans_23082020.xlsx')

dbscan_loaded_model = pickle.load(open(dbscan_Model, 'rb'))

pred = dbscan_loaded_model.predict(X1)
dbscan_out = data
dbscan_out['predicted'] = list(pred)
dbscan_out.to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\actual\final_dbscan_23082020.xlsx')
'''
ann_loaded_model = pickle.load(open(ann_Model, 'rb'))

pred = ann_loaded_model.predict(X)
ann_out = data
ann_out['predicted'] = list(pred)
ann_out.to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\actual\final_dbscan_23082020.xlsx')


