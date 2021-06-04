import pandas as pd
import pickle
from Encoding import LabelEncoding
from DataPreprocessing.scaler import ScaleTransform
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import Master.MasterAlgorithm as master

def wrapperFunction(dataframe,attributeDictionary,dimensionColumn,wrapperModelCombinations):
    dimensions = tuple(set(dataframe[dimensionColumn]))
    categoryModelPairs = dict.fromkeys(dimensions,wrapperModelCombinations)
    categories = {}
    accuracyDictionary = {}

    finalPredictions = []
    for category in dimensions:
        y_pred = []
        print('Category : ', category)
        df_name = category + '_DF'  # the name for the dataframe

        categories[df_name] = dataframe.loc[dataframe[dimensionColumn] == category, dataframe.columns]
        # print(df_name)
        print('Number of rows in the category ', category, ' are :', ((categories[df_name]).shape)[0])
        # print((categories[df_name]).shape)

        X = (categories[df_name]).iloc[:, :-1].values
        y = (categories[df_name]).iloc[:, -1].values

        X = LabelEncoding.LabelEncode(X)
        X = ScaleTransform(X, 'standard')

        print('Assigned Models are : ', categoryModelPairs[category])

        categoryAccuracyDict = dict.fromkeys(categoryModelPairs[category])
        for savedModel in categoryModelPairs[category]:

            #loadedModel = pickle.load(open(savedModel, 'rb'))

            #y_pred = loadedModel.predict(X)

            attributeDictionary[custom] = savedModel
            outData = master.findCombination(attributeDictionary)
            y_pred = outData['predicted']

            finalPredictions.extend(y_pred)


            print('############ Classification Report : ', df_name, ' #################')

            y_test = list(y)
            cm = confusion_matrix(y_test, y_pred)
            accuracy = (accuracy_score(y_test, y_pred)) * 100
            categoryAccuracyDict[savedModel] = accuracy

            print('Accuracy :' + str(accuracy) + '%')
            print(' Confusion Matrix : ' + str(cm))

            classificationReport = classification_report(y_test, y_pred)
            print('Classification details :')
            print(classificationReport)

        accuracyDictionary[category] = categoryAccuracyDict

    selectedModelForCategory = dict.fromkeys(dimensions)
    for category in accuracyDictionary.keys():
        modelAccuracyList = (accuracyDictionary[category]).values
        modelAccuracyList.sort(key=lambda x : x[1],reversed=True)
        selectedModelForCategory[category] = (modelAccuracyList[0])[0]


if __name__ == '__main__':

    dataframe = None

    ticketId = ''
    trainingColumns = None
    outputColumn = None
    dataframe = None
    replaceMissing = True
    dropCardinal = True
    ifPCA = True
    pcaComp = 3
    X_train = 'null'
    X_test = 'null'
    y_train = 'null'
    y_test = 'null'
    inputType = 'labelled'
    contentType = 'text'
    mlType = 'supervised'
    encodingType = 'one_hot'
    scalingType = 'standard'
    usecaseType = 'classification'
    custom = ''
    numClusters = 2
    epsilon = 0.3
    minSamples = 10
    trainedInput = 3
    trainedOut = 1
    numClasses = 2
    bandwidth = 2
    transactionIdColumn = ''
    ItemsColumn = ''
    cList = []
    gammaList = []
    kernelList = []
    leavesList = []
    alphaList = []
    minDataInLeafList = []
    maxDepthList = []
    boostingTypeList = []
    learningRateList = []
    estimatorsList = []
    maxFeaturesList = []

    minSamplesSplitList = []
    minSamplesLeafList = []
    bootStrapList = []
    SL = None
    size = None
    randomstate = None
    optimization = False
    CrossValidation = None
    cv = None
    n_jobs = None
    max_iter = []
    polynomialDegree = 2
    numNonZeroCoeffs = 1

    masterDictionary: {ticketId: ticketId, trainingColumns: trainingColumns, outputColumn: outputColumn, replaceMissing: replaceMissing,
                       dropCardinal: dropCardinal, ifPCA: True, pcaComp: 3, X_train: 'null', X_test: 'null', y_train: 'null',
                       y_test: 'null', inputType: 'labelled', contentType: 'text', mlType: 'supervised',
                       encodingType: 'one_hot',
                       scalingType: 'standard', usecaseType: 'classification', custom: '', numClusters: 2, epsilon: 0.3,
                       minSamples: 10, trainedInput: 3, trainedOut: 1, numClasses: 2, bandwidth: 2,
                       transactionIdColumn: '', ItemsColumn: '',
                       cList: [], gammaList: [], kernelList: [], leavesList: [], alphaList: [], minDataInLeafList: [],
                       maxDepthList: [], boostingTypeList: [], learningRateList: [], estimatorsList: [],
                       maxFeaturesList: [], minSamplesSplitList: [], minSamplesLeafList: [], bootStrapList: [],
                       SL: None, size: None, randomstate: None, optimization: False, CrossValidation: None, cv: None,
                       n_jobs: None,
                       max_iter: [], polynomialDegree: 2, numNonZeroCoeffs: 1,custom : ''}

