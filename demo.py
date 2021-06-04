import pandas as pd
import Master.MasterAlgorithm as master
import Master.masterVisualisation as visualise
from sklearn.model_selection import train_test_split
import numpy as np

from Encoding import LabelEncoding
from DataPreprocessing.scaler import ScaleTransform
import pickle



data = pd.read_excel(r"D:\Personal\SmartIT\data\Credit Card Fraud\train.xlsx")


################################################################################################################
"""1.0 Reading custom inputs from user, to be read in from UI. """
################################################################################################################

dataPath = r"D:\Personal\SmartIT\data\Credit Card Fraud\train.xlsx"
data = pd.read_excel(r"D:\Personal\SmartIT\data\Credit Card Fraud\train.xlsx")
#chartsList = ['barPlot','boxenPlot','cluster','heat','scatter','pairGrid']
chartsList = ['barPlot','pairGrid']
visualisationAttributeDict = {'ticketId':'23082020','x':"V1", 'y':"Class", 'hue':"Time",'category':"Amount"}


################################################################################################################
"""1.1 Visualisation of data based on user's input. """
################################################################################################################

visualise.chartPlotter(dataPath=dataPath,data=data,chartsList=chartsList,inputDict=visualisationAttributeDict,splitPercentage=1,outputPath=r'D:\Personal\SmartIT\data\walkthrough\demo\training',dataType='train')
'''

##############################################################################################################
'''
trainingColumns = (list(data.columns))[:-1]
print('training col :', trainingColumns)
outputColumn = (list(data.columns))[-1]
print('output column :', outputColumn)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=42)

XgBoost_out = master.findCombination(ticketId='23082020_XgBoost',trainingColumns=trainingColumns,outputColumn=outputColumn
                             ,dataframe=X_test,X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test,numClasses=2,
                             inputType = 'labelled',mlType='supervised',contentType = 'text',usecaseType = 'boosting',encodingType ='label',scalingType ='standard',custom ='xgboost')
(XgBoost_out[0]).to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_XgBoost_23082020.xlsx')


################################################################################################################
data = pd.read_excel(r"D:\Personal\SmartIT\data\Credit Card Fraud\train.xlsx")

trainingColumns = (list(data.columns))[:-1]
print('training col :', trainingColumns)
outputColumn = (list(data.columns))[-1]
print('output column :', outputColumn)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print('Xdata type :', type(X))
print('ydata type :', type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)

estimatorsList = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 100)]
maxFeaturesList = [0.02, 0.03]
maxDepthList = ['auto', 'sqrt']
maxDepthList = [int(x) for x in np.linspace(10, 110, num = 11)]
maxDepthList.append(None)
minSamplesSplitList = [2, 5, 10]
minSamplesLeafList = [1, 2, 4]
bootStrapList = [True, False]

out = master.findCombination(ticketId='28082020_RandomForest',trainingColumns=trainingColumns,outputColumn=outputColumn
                             ,dataframe=X_test,X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test,numClasses=2,
                             estimatorsList=[], maxFeaturesList=[], maxDepthList=[], minSamplesSplitList=[], minSamplesLeafList=[],
                             bootStrapList=[],
                             inputType = 'labelled',mlType='supervised',contentType = 'text',usecaseType = 'boosting',encodingType ='label',scalingType ='standard',custom ='randomForest')
(out[0]).to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_randomForest_23082020.xlsx')


################################################################################################################
data = pd.read_excel(r"D:\Personal\SmartIT\data\Credit Card Fraud\train.xlsx")

trainingColumns = (list(data.columns))[:-1]
print('training col :', trainingColumns)
outputColumn = (list(data.columns))[-1]
print('output column :', outputColumn)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print('Xdata type :', type(X))
print('ydata type :', type(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=42)
leavesList = [2000, 2500]
alphaList = [0.02, 0.03]
minDataInLeafList = [5, 10]
maxDepthList = [5, 10]
boostingTypeList = ['gbdt']
learningRateList = [0.001,0.002,0.003]
out = master.findCombination(ticketId='23082020_LGBM',trainingColumns=trainingColumns,outputColumn=outputColumn,leavesList=[],alphaList=[],minDataInLeafList=[]
                             ,maxDepthList=[],boostingTypeList=[],learningRateList=[]
                             ,dataframe=X_test,X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test,numClasses=2,inputType = 'labelled',mlType='supervised',contentType = 'text',usecaseType = 'boosting',encodingType ='label',scalingType ='standard',custom ='lgbm')
(out[0]).to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_lgbm_23082020.xlsx')



kmeans_out =  master.findCombination(ticketId='23082020'
                             ,dataframe=data,X_train=data,numClusters=2,inputType = 'unlabelled',mlType='unsupervised',contentType = 'text',usecaseType = 'clustering',encodingType = 'label',custom = 'kmeans')
(kmeans_out[0]).to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_kmeans_23082020.xlsx')


dbscan_out = master.findCombination(ticketId='123'
                                  , dataframe=data, X_train=data,epsilon=0.01,minSamples=2,inputType = 'unlabelled',mlType='unsupervised',contentType = 'text' , usecaseType = 'clustering' , encodingType = 'label' , scalingType = 'standard' , custom = 'dbscan'  )
(dbscan_out[0]).to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_dbscan_23082020.xlsx')




ann_out = master.findCombination(dataframe=data, inputType='labelled', mlType='deepLearning',usecaseType = 'classification',
                                 contentType='text', custom='ann',ticketId='23082020')
(ann_out[0]).to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_ann_23082020.xlsx')


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
ann_Model = r'D:\Personal\SmartIT\DataScience_Latest\ann_.sav'

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

dataPath = r"D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_kmeans_23082020.xlsx"
data = pd.read_excel(r"D:\Personal\SmartIT\data\Credit Card Fraud\train.xlsx")
#chartsList = ['barPlot','boxenPlot','cluster','heat','scatter','pairGrid']
chartsList = ['cluster']
visualisationAttributeDict = {'ticketId':'23082020','x':"V1", 'y':"predicted", 'hue':"Time",'category':"Amount"}


################################################################################################################
"""1.1 Visualisation of data based on user's input. """
################################################################################################################

visualise.chartPlotter(dataPath=dataPath,data=data,chartsList=chartsList,inputDict=visualisationAttributeDict,splitPercentage=1,outputPath=r'D:\Personal\SmartIT\data\walkthrough\demo\predicted',dataType='predict')

