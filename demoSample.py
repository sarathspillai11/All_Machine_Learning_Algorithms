import pandas as pd
from Master import MasterAlgorithm as master
from Master import masterVisualisation as visualise
from sklearn.model_selection import train_test_split
import numpy as np

################################################################################################################
"""1.0 Reading custom inputs from user, to be read in from UI. """
################################################################################################################

dataPath = r"D:\Personal\SmartIT\data\Credit Card Fraud\train.xlsx"
data = pd.read_excel(r"D:\Personal\SmartIT\data\Credit Card Fraud\train.xlsx")
chartsList = ['barPlot','boxenPlot','cluster','heat','scatter','pairGrid']
visualisationAttributeDict = {'ticketId':'23082020','x':"V1", 'y':"Class", 'hue':"Time",'category':"Amount"}


################################################################################################################
"""1.1 Visualisation of data based on user's input. """
################################################################################################################

visualise.chartPlotter(dataPath=dataPath,data=data,chartsList=chartsList,inputDict=visualisationAttributeDict,splitPercentage=90,outputPath=r'D:\Personal\SmartIT\data\walkthrough\demo\training',dataType='train')


################################################################################################################
"""2.0 Reading custom input from user for  training . """
################################################################################################################



################################################################################################################
"""2.1 Execute user's selection of algorithms for training.

User selects XgBoost, LightGBM and Random Forest algorithms.
"""
################################################################################################################


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


###############################################################################################################

################################################################################################################
# """1.0 Reading custom inputs from user, to be read in from UI. """
# ################################################################################################################
#
dataPath = r':\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_randomForest_23082020.xlsx'
data = pd.read_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_randomForest_23082020.xlsx')


chartsList = ['barPlot','boxenPlot','cluster','heat','scatter','pairGrid']
visualisationAttributeDict = {'ticketId':'23082020_RandomForest','x':"V1", 'y':"Class", 'hue':"Time",'category':"Amount"}

#
# ################################################################################################################
"""1.1 Visualisation of data based on user's input. """
# ################################################################################################################

visualise.chartPlotter(dataPath=dataPath,data=data,chartsList=chartsList,inputDict=visualisationAttributeDict,splitPercentage=90,outputPath=r'D:\Personal\SmartIT\data\walkthrough\demo\predicted',dataType='predicted_randomForest')

###################################################################################################################

dataPath = r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_lgbm_23082020.xlsx'
data = pd.read_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_lgbm_23082020.xlsx')




chartsList = ['barPlot','violin','divLolly']
visualisationAttributeDict = {'ticketId':'23082020_LGBM','x':"V1", 'y':"Class", 'hue':"Time",'category':"Amount"}

#
# ################################################################################################################
"""1.1 Visualisation of data based on user's input. """
# ################################################################################################################

visualise.chartPlotter(dataPath=dataPath,data=data,chartsList=chartsList,inputDict=visualisationAttributeDict,splitPercentage=90,outputPath=r'D:\Personal\SmartIT\data\walkthrough\demo\predicted',dataType='predicted_LightGBM')

###################################################################################################################

dataPath = r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_XgBoost_23082020.xlsx'
data = pd.read_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_XgBoost_23082020.xlsx')



chartsList = ['barPlot','boxenPlot','cluster','heat','scatter','pairGrid']
visualisationAttributeDict = {'ticketId':'23082020_xgBoost','x':"V1", 'y':"Class", 'hue':"Time",'category':"Amount"}

#
# ################################################################################################################
"""1.1 Visualisation of data based on user's input. """
# ################################################################################################################

visualise.chartPlotter(dataPath=dataPath,data=data,chartsList=chartsList,inputDict=visualisationAttributeDict,splitPercentage=90,outputPath=r'D:\Personal\SmartIT\data\walkthrough\demo\predicted',dataType='predicted_XgBoost')