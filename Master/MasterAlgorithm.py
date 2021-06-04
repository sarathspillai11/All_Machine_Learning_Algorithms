from Bayesian import BayesianNetworkPrediction
import Clustering.ExpectationMaximisation as em, Clustering.dBScanClustering as db, Clustering.HierarchichalClusteringPredictor as hierarchical, Clustering.kMediansClustering as kmed, Clustering.kmeans as km
import Clustering.SpectralClustering as spec, Clustering.meanShift as mean, Clustering.affinityPropagation as aff
from DecisionTree.IterativeDichotomiser3 import prediction_dt_id3,dt_id3
#from DecisionTree.Chi import Chi
import Regression.Linear as Linear
from DecisionTree import conditional_decision_tree
from Encoding import LabelEncoding
import NeuralNetworks.HopfieldNetworkPredictor as hope,NeuralNetworks.PerceptronClassifier as perceptron
from Regression import MultivariateAdaptiveRegressionSplines,logisticRegression,poly_features_regression,Least_angle_regression
from Classification import decisiontree,gaussian_naive_baise,gradient_boost_classifier,k_nearest_neighbour,SVMClassifier,MultinomialNaive
from Boosting import random_forest,XgBoost,lightGBMClassifier
#Least_angle_regression,poly_features_regression,logistic_regression
from Regularization import ElasticNet,lasso,ridge
from Scaling import StandardScaling
import Vectorization.tfIdfVectorization as tfidf
import AssociationRuleLearning.AprioriRecommender as ap,AssociationRuleLearning.EclatRecommender as ec
from DataPreprocessing.scaler import ScaleTransform
import re,os
from Classification.Learning_Vector_Quantization import *
from DecisionTree.DecisionStump import *
from DecisionTree import ClassificationAndRegressionTree as cart
#import DeepLearning.LSTMSentencePrediction as ls, DeepLearning.cnnPredictor as cnn, DeepLearning.DeepBeliefNetwork as rbm, DeepLearning.ArtificialNeuralNetworksClassifier as ann, DeepLearning.linear_regression as lr, DeepLearning.autoencoder_torch as auto, DeepLearning.logisticRegression as logi, DeepLearning.RadialBasisFunctionNetWorkClassifier as rbfn
#import DeepLearning.autoencoder_torch as torchAuto, DeepLearning.autoencoderKerasImages as encoder, DeepLearning.restricted_boltzmann_machine as bolt, DeepLearning.self_organized_maps as som, DeepLearning.support_vector_regression as svr, DeepLearning.cNN_ImageClassifier as CNNImage
import DeepLearning.RadialBasisFunctionNetWorkClassifier as rbfn
import DeepLearning.ArtificialNeuralNetworksClassifier as ann
import pandas as pd
from LogMaster import Logger
import traceback
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from Encoding import inverseTransformData as invTransformer
from sklearn.metrics import classification_report
from DataPreprocessing import missingvalues
# import Deep_Learning_Usecases.ageGroupDetectionKerasMaster.ageGroupDetectionTrain as ageDetect
# import Deep_Learning_Usecases.ChildVsAdultDetectionKerasMaster.childAdultTraining as childAdult
# import Deep_Learning_Usecases.genderDetectionKerasMaster.genderDetectionTraining as genderDetect
# import Deep_Learning_Usecases.raceDetectionKerasMaster.raceDetectionTraining as raceDetect
# import  Deep_Learning_Usecases.RealtimeEmotionDetectionMaster.emotionRecognition as emotionDetect
# import Deep_Learning_Usecases.RealtimeMotionDetection.MotionDetection as motionDetect
from Master import masterVisualisation as visualise
from MediaProcessing.MediaComparison import  mediaCompare as media_compare

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def findCombination(ticketId='',trainingColumns=None,outputColumn=None,dataframe=None,replaceMissing=True,
                    dropCardinal=True,ifPCA=True,pcaComp=3,X_train= 'null', X_test= 'null', y_train = 'null',
                    y_test = 'null',inputType = 'labelled',contentType='text',mlType='supervised',encodingType='one_hot',
                    scalingType='standard',usecaseType='classification',custom='',numClusters=2,epsilon=0.3,
                    minSamples=10,trainedInput=3,trainedOut = 1,numClasses=2,bandwidth=2,transactionIdColumn='', ItemsColumn='',
                    cList = [] , gammaList = [], kernelList= [],leavesList=[],alphaList=[],minDataInLeafList=[],
                    maxDepthList=[],boostingTypeList=[],learningRateList=[],estimatorsList=[],maxFeaturesList=[]
                    ,minSamplesSplitList=[],minSamplesLeafList=[],bootStrapList=[],
                    SL=None, size=None, randomstate=None, optimization=False,CrossValidation=None , cv=None, n_jobs=None,
                    max_iter=[],polynomialDegree=2,numNonZeroCoeffs=1,mediaSourcePath='',mediaLookupPath='',mediaSourceType='',
                    mediaLookupType='',compareTextOrImage=False,textFromImageFlag=True, analysisTypes=['pattern'],
                    isolationForestFlag=False,isolationForestContamination=0.1,ellipticEnvelopeFlag=False,ellipticEnvelopeContamination=0.1,
                    localOutlierFlag=False,oneClassSVMFlag=False,oneClassOutlierRatio=0.01):
    print('inside find combination')

    if(usecaseType=='mediaProcessing'):
        media_compare(sourcePath=mediaSourcePath, lookupPath=mediaLookupPath, sourceType=mediaSourceType, lookupType=mediaLookupType, compareTextorImage=compareTextOrImage,textFromImageFlag=textFromImageFlag, analysisTypes=analysisTypes)

    log = Logger.logger
    log.info('logging started')
    try:
        log_data = pd.DataFrame(columns=['CombinationID','StepNo','Input','Output','StepDescription','Info'])

        print('########### type of dataframe at start  : ',type(dataframe))
        if(isolationForestFlag):
            # identify outliers in the training dataset
            iso = IsolationForest(contamination=isolationForestContamination)
            yhat = iso.fit_predict(X_train)
            # select all rows that are not outliers
            mask = yhat != -1
            X_train, y_train = X_train[mask, :], y_train[mask]  # summarize the shape of the updated training datas

        if(ellipticEnvelopeFlag):
            # identify outliers in the training dataset
            ee = EllipticEnvelope(contamination=ellipticEnvelopeContamination)
            yhat = ee.fit_predict(X_train)
            # select all rows that are not outliers
            mask = yhat != -1
            X_train, y_train = X_train[mask, :], y_train[mask]
        if(localOutlierFlag):
            # identify outliers in the training dataset
            lof = LocalOutlierFactor()
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = yhat != -1
            X_train, y_train = X_train[mask, :], y_train[mask]
        if(oneClassSVMFlag):
            ee = OneClassSVM(nu=oneClassOutlierRatio)
            yhat = ee.fit_predict(X_train)
            # select all rows that are not outliers
            mask = yhat != -1
            X_train, y_train = X_train[mask, :], y_train[mask]
        if (replaceMissing and mlType == 'supervised'):
            X_train = missingvalues.replacemissingvalue(X_train)
            X_test = missingvalues.replacemissingvalue(X_test)
        # if(dropCardinal): X_train=data_gaurdling_methods.drop_highcardinality(X_train)
        # if(ifPCA) : X_train = principal_component_analysis(X_train,pcaComp)

        if(mlType != 'unsupervised'):
            try:
                outData = pd.concat([X_test, y_test], axis=1)
            except:
                outData = X_test
            if(custom == 'ann'):
                pass
            else:
                dataframe = X_test
            if(custom != 'id3' and custom != 'ann'):
                X_train = X_train.values
                y_train = y_train.values
                X_test = X_test.values
            try:
                y_test = y_test.values
            except:
                log.error('y_test is not available to check accuracy')
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
            log.info('preprocessing completed !!!')

        else:
            outData = dataframe
            dataframe = dataframe.values
        print('########### type of dataframe at mid  : ', type(dataframe))
        if(inputType == 'labelled' and mlType == 'supervised'):
            print('labelled data that needs supervised approach')
            log.info('labelled data that needs supervised approach')
            if(contentType == 'text' and usecaseType == 'regression' and encodingType =='label' and scalingType =='standard' and custom =='mars'):
                print('inside mars in master program')
                X_train, X_test = LabelEncoding.LabelEncode(X_train),LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                #X_train = count.countVectorize(dataframe)
                print('encoding and scaling completed')
                dataframe = MultivariateAdaptiveRegressionSplines.MarsImplementation(dataframe,X_train,y_train,X_test,ticketId)
                log_data['CombinationID'] = ['C1'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding','Scaling','MARS Regression']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = ['Converting the labels into numeric form so as to convert it into the machine-readable form',
                                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                                    'Multi variate adaptive regression splines. The output model have been saved at the location'+str(os.getcwd())+os.sep+'mars_' + ticketId + '.sav']
            elif(contentType == 'text' and usecaseType == 'regression' and encodingType =='label' and scalingType =='standard' and custom =='linear'):
                print('flowed to linear in master')


                X_train, X_test, = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                print('linear encoding completed')
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                print('just before actual func call')
                dataframe = Linear.linearRegressor(dataframe, X_train, y_train, X_test, SL=SL, size=size, randomstate= randomstate , optimization= optimization,ticketId = ticketId)
                log_data['CombinationID'] = ['C2'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Linear Regression']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range.',
                    'Linear Regression.  The output model have been saved at the location'+str(os.getcwd())+os.sep+'linearRegression_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'regression' and encodingType == 'label' and scalingType == 'standard' and custom == 'logistic'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = logisticRegression.logistic_regression(dataframe, X_train, y_train, X_test,optimization = optimization, CrossValidation = CrossValidation,
                                                                   cv = cv, n_jobs = n_jobs,max_iter=max_iter, ticketId = ticketId)

                log_data['CombinationID'] = ['C3'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Linear Regression']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range.',
                    'Logistic Regression. The output model have been saved at the location'+str(os.getcwd())+os.sep+'logisticRegression_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'regression' and encodingType == 'label' and scalingType == 'standard' and custom == 'poly'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = poly_features_regression.polynomial_feature_regression(dataframe, X_train, y_train, X_test,polynomialDegree,ticketId)
                log_data['CombinationID'] = ['C4'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Linear Regression']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range.',
                    'Polynomial Regression. The output model have been saved at the location'+str(os.getcwd())+os.sep+'poly_regression_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'regression' and encodingType == 'label' and scalingType == 'standard' and custom == 'lar'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = Least_angle_regression.least_angle_regression(dataframe, X_train, y_train, X_test,numNonZeroCoeffs,ticketId)
                log_data['CombinationID'] = ['C4'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Linear Regression']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range.',
                    'Least Angle Regression. The output model have been saved at the location'+str(os.getcwd())+os.sep+'lar_' + ticketId + '.sav']
            elif (contentType == 'integer' and usecaseType == 'regularization' and encodingType == 'label' and scalingType == 'standard' and custom == 'elastic'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                X_train = tfidf.tfIdfVectorize(dataframe)
                dataframe = ElasticNet.ElasticNetPredictor(dataframe,X_train,y_train,X_test,ticketId)
                log_data['CombinationID'] = ['C5'] * 4
                log_data['StepNo'] = list(range(1, 5))
                log_data['StepDescription'] = ['Encoding', 'Scaling','TF IDF Vectorisation','Linear Regression']
                log_data['Input'] = [trainingColumns] * 4
                log_data['Output'] = [outputColumn] * 4
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'Converts the text into vectors on the basis of probability of occurrence of a word in a document with respect to that in other documents',
                    'Elastic Net Regularisation. The output model have been saved at the location'+str(os.getcwd())+os.sep+'elasticNet_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'regularization' and encodingType == 'label' and scalingType == 'standard' and custom == 'ridge'):
                print('in ridge inside master')
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train, scalingType), ScaleTransform(X_test, scalingType)
                # X_train = tfidf.tfIdfVectorize(dataframe)
                dataframe = ridge.ridgeRegression(dataframe, X_train, y_train, X_test, ticketId)
                log_data['CombinationID'] = ['C6'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Ridge Regression']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'Elastic Net Regularisation. The output model have been saved at the location' + str(
                        os.getcwd()) + os.sep + 'elasticNet_' + ticketId + '.sav']
            elif(contentType == 'text' and usecaseType == 'regularization' and encodingType =='label' and scalingType =='standard' and custom == 'elastic'):
                print('in elastic for text')
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                #X_train = tfidf.tfIdfVectorize(dataframe)
                dataframe = ElasticNet.ElasticNetPredictor(dataframe,X_train,y_train,X_test,ticketId)
                log_data['CombinationID'] = ['C6'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Elastic Net Regression']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'Elastic Net Regularisation. The output model have been saved at the location'+str(os.getcwd())+os.sep+'elasticNet_' + ticketId + '.sav']
            elif(contentType == 'text' and usecaseType == 'regularization' and encodingType == 'label' and scalingType == 'standard' and custom == 'lasso'):
                print('in lasso for text')
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                # X_train = tfidf.tfIdfVectorize(dataframe)

                dataframe = lasso.lasso_(dataframe, X_train, y_train, X_test,ticketId)
                log_data['CombinationID'] = ['C7'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Lasso Regression']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'Lasso Regression. The output model have been saved at the location'+str(os.getcwd())+os.sep+'lasso_' + ticketId + '.sav']

            elif (contentType == 'text' and usecaseType == 'regression' and custom == 'torchSVR'):
                print('in elastic for text')
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train, scalingType), ScaleTransform(X_test, scalingType)
                dataframe = svr.supportvectorregression(dataframe, X_train, y_train, X_test,trainedInput,trainedOut,ticketId)
                log_data['CombinationID'] = ['C8'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Support Vector Regression']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'SVR using pytorch. The output model have been saved at the location'+str(os.getcwd())+os.sep+'torchSVR_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'binary_classification' and encodingType == 'label' and scalingType == 'standard' and custom == 'guassian'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = gaussian_naive_baise.gaussian_naive(dataframe, X_train, y_train, X_test,ticketId,numClasses)
                log_data['CombinationID'] = ['C9'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Guassian naive bayes classification']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'gaussian_naive_baise. The output model have been saved at the location'+str(os.getcwd())+os.sep+'guassian_naive_' + ticketId + '.sav']

            elif (contentType == 'text' and usecaseType == 'classification' and encodingType == 'label' and scalingType == 'standard' and custom == 'naive'):
                print('inisde naive in master')
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = MultinomialNaive.multinomialNaiveClassifier(dataframe, X_train, y_train, X_test,ticketId)
                log_data['CombinationID'] = ['C9'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Multinomial naive bayes classification']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'Multinomial Naive Bayes. The output model have been saved at the location'+str(os.getcwd())+os.sep+'naiveBayes_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'classification' and encodingType == 'label' and scalingType == 'standard' and custom == 'gradient'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = gradient_boost_classifier.gradient_boost_classfier(dataframe, X_train, y_train, X_test,ticketId)
                log_data['CombinationID'] = ['C10'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Gradient Boost classification']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'Gradient Boost. The output model have been saved at the location'+str(os.getcwd())+os.sep+'gradient_boost_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'classification' and encodingType == 'label' and scalingType == 'standard' and custom == 'knn'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = k_nearest_neighbour.knn(dataframe, X_train, y_train, X_test,ticketId)
                log_data['CombinationID'] = ['C11'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'knn classification']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'K- nearest neighbours. The output model have been saved at the location'+str(os.getcwd())+os.sep+'knn_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'boosting' and encodingType == 'label' and scalingType == 'standard' and custom == 'randomForest'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                print(X_train)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = random_forest.random_forest(dataframe, X_train, y_train, X_test,ticketId,estimatorsList=estimatorsList, maxFeaturesList=maxFeaturesList,
                                                        maxDepthList=maxDepthList, minSamplesSplitList=minSamplesSplitList, minSamplesLeafList=minSamplesLeafList,
                                 bootStrapList=bootStrapList,numClasses=numClasses)
                log_data['CombinationID'] = ['C12'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'random Forest classification']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'random forest. The output model have been saved at the location'+str(os.getcwd())+os.sep+'randomForest_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'classification' and encodingType == 'label' and scalingType == 'standard' and custom == 'decisionTree'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = decisiontree.decision_tree(dataframe, X_train, y_train, X_test,ticketId)
                log_data['CombinationID'] = ['C13'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'decision tree classification']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'decision tree. The output model have been saved at the location'+str(os.getcwd())+os.sep+'decision_tree_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'classification' and encodingType == 'label' and scalingType == 'standard' and custom == 'svc'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = SVMClassifier.svmClassification(dataframe, X_train, y_train, X_test,ticketId,cList, gammaList, kernelList)
                log_data['CombinationID'] = ['C14'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'svm classification']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'svm. The output model have been saved at the location'+str(os.getcwd())+os.sep+'svm_' + ticketId + '.sav']

            elif (contentType == 'text' and usecaseType == 'classification' and encodingType == 'label' and scalingType == 'standard' and custom == 'conditionalDecision'):
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train,scalingType),ScaleTransform(X_test,scalingType)
                dataframe = conditional_decision_tree.con_dec_tree(dataframe, X_train, y_train, X_test,ticketId)
                log_data['CombinationID'] = ['C15'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'conditional Decision tree classification']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'conditional decision tree. The output model have been saved at the location'+str(os.getcwd())+os.sep+'conditional_decision_tree_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'bayesian' and encodingType == 'label' and scalingType == 'standard' and custom == 'bayesian'):
                print('inside bayesian inside master')
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train, scalingType), ScaleTransform(X_test, scalingType)
                dataframe = BayesianNetworkPrediction.BayesianNetworkPredictor(dataframe,X_train,y_train,X_test,ticketId)
                log_data['CombinationID'] = ['C16'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Bayesian classification']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'Bayesian Network. The output model have been saved at the location'+str(os.getcwd())+os.sep+'Bayesian_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'bayesian' and encodingType == 'one-hot' and scalingType == 'standard'):
                # change needed here
                dataframe = LabelEncoding(dataframe)
                dataframe = StandardScaling(dataframe)
                dataframe = BayesianNetworkPrediction.BayesianNetworkPredictor(X_train, X_test, y_train,ticketId)
            elif (contentType == 'text' and usecaseType == 'decisionTree' and encodingType == 'label' and scalingType == 'standard' and custom == 'id3'):
                # X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                # X_train, X_test = ScaleTransform(X_train, scalingType), ScaleTransform(X_test, scalingType)
                # #dataframe = prediction_dt_id3(X_train, X_test, y_train)
                # print('Xdata type :', type(X_train))
                # print('ydata type :', type(y_train))
                dt_model = dt_id3(Xdata=X_train, ydata=y_train, pre_pruning="chi_2", chi_lim=0.1)
                dataframe = prediction_dt_id3(dt_model, X_test)
                y_pred = [len(i) for i in dataframe['predicted']]
                dataframe['predicted'] = y_pred
                log_data['CombinationID'] = ['C17'] * 4
                log_data['StepNo'] = list(range(1, 5))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Data Model Creation','ID3']
                log_data['Input'] = [trainingColumns] * 4
                log_data['Output'] = [outputColumn] * 4
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'Data Model creation',
                    'Iterative Dichotomiser. The output model have been saved at the location'+str(os.getcwd())+os.sep+'id3_' + ticketId + '.sav']

            elif (contentType == 'text' and usecaseType == 'bayesian' and encodingType == 'label' and scalingType == 'standard' and custom == 'chiSquare'):
                dataframe = LabelEncoding(dataframe)
                dataframe = StandardScaling(dataframe)
                # change needed here
                #dataframe = Chi(X_train, X_test, y_train)
                log_data['CombinationID'] = ['C18'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'Chi Square Test']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'Chi Square Test. The output model have been saved at the location'+str(os.getcwd())+os.sep+'Chi_' + ticketId + '.sav']


            elif (contentType == 'text' and usecaseType == 'boosting' and encodingType == 'label' and scalingType == 'standard' and custom == 'xgboost'):
                print('inside xgboost in master')
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train, scalingType), ScaleTransform(X_test, scalingType)

                dataframe = XgBoost.XgBoostPredictor(dataframe, X_train, y_train, X_test,ticketId,numClasses=numClasses)

                log_data['CombinationID'] = ['C21'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'XgBoost']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'XgBoost Classifier. The output model have been saved at the location'+str(os.getcwd())+os.sep+'XgBoost_' + ticketId + '.sav']

            elif (contentType == 'text' and usecaseType == 'boosting' and encodingType == 'label' and scalingType == 'standard' and custom == 'lgbm'):
                X_train, X_test  = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)

                print('y_train values before encoding :',set(list(y_train)))
                y_train,encoderObj = LabelEncoding.LabelEncode(y_train)
                print('y_train values after encoding :',set(list(y_train)))
                X_train, X_test  = ScaleTransform(X_train, scalingType), ScaleTransform(X_test, scalingType)
                #y_train = ScaleTransform(y_train,scalingType)
                dataframe = lightGBMClassifier.lightGBMPredictor(dataframe, X_train, y_train, X_test,ticketId,leavesList,alphaList,minDataInLeafList,maxDepthList,boostingTypeList,learningRateList,numClasses)
                dataframe['predicted'] = encoderObj.inverse_transform(dataframe['predicted'])
                log_data['CombinationID'] = ['C22'] * 3
                log_data['StepNo'] = list(range(1, 4))
                log_data['StepDescription'] = ['Encoding', 'Scaling', 'LightGBM']
                log_data['Input'] = [trainingColumns] * 3
                log_data['Output'] = [outputColumn] * 3
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. ',
                    'LightGBM Classifier. The output model have been saved at the location'+str(os.getcwd())+os.sep+'LightGBM_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'classification' and custom == 'cart'):
                # Test CART on Bank Note dataset
                # Test CART on Bank Note dataset
                seed(1)
                # load and prepare data

                # convert string attributes to integers
                for i in range(len(dataframe[0])):
                    str_column_to_float(dataframe, i)
                # evaluate algorithm
                n_folds = 5
                max_depth = 5
                min_size = 10
                scores = evaluate_algorithm(dataframe, cart.decision_tree, n_folds, max_depth, min_size)
                print('Scores: %s' % scores)
                print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
                log_data['CombinationID'] = ['C23']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['CART Classification']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Conditional and Regression Tree. The output model have been saved at the location'+str(os.getcwd())+os.sep+'cart_' + ticketId + '.sav']
            elif(contentType=='text' and usecaseType=='classification' and custom=='lvq'):
                print('insdide lvq in master')
                seed(1)
                # load and prepare data

                for i in range(len(dataframe[0]) - 1):
                    str_column_to_float(dataframe, i)
                # convert class column to integers
                str_column_to_int(dataframe, len(dataframe[0]) - 1)
                # evaluate algorithm
                n_folds = 5
                learn_rate = 0.3
                n_epochs = 50
                n_codebooks = 20
                scores = evaluate_algorithm(ticketId,dataframe, learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
                print('Scores: %s' % scores)
                print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
                log_data['CombinationID'] = ['C24']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['LVQ']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Learning Vector Qunatization. The output model have been saved at the location'+str(os.getcwd())+os.sep+'LVQ_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'classification' and custom == 'stump'):

                datMat, classLabels = dataframe.load_wine(True)
                print(datMat.shape[0])
                D = mat(ones((datMat.shape[0], 1)) / datMat.shape[0])
                bestStump, minError, bestClasEst = buildStump(datMat, classLabels, D,ticketId)
                print(bestStump, minError, bestClasEst)
                log_data['CombinationID'] = ['C25']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['Decision Stump']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Decision Stump Classifier. The output model have been saved at the location'+str(os.getcwd())+os.sep+'decision_stump_' + ticketId + '.sav']

            elif (contentType == 'image' and usecaseType == 'neural' and encodingType == 'label' and scalingType == 'standard' and custom == 'hopfield'):
                # First, you can create a list of input file path
                current_path = r'D:\Personal\SmartIT\data'
                train_paths = []
                path = r"D:\Personal\SmartIT\data\hopfield\train" + os.sep
                for i in os.listdir(path):
                    if re.match(r'[0-9a-zA-Z-]*.jp[e]*g', i):
                        train_paths.append(path + i)
                # Second, you can create a list of sungallses file path
                test_paths = []
                path = r"D:\Personal\SmartIT\data\hopfield\test" + os.sep
                for i in os.listdir(path):
                    if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g', i):
                        test_paths.append(path + i)
                print('in here for hope master')
                hope.hopfield(train_files=train_paths, test_files=test_paths, theta=0.5,time=20000,size=(100,100),threshold=60, current_path = current_path)
                log_data['CombinationID'] = ['C26']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['Hopfield Network']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Hopfield Network. The output model have been saved at the location'+str(os.getcwd())+os.sep+'hopfield_' + ticketId + '.sav']
            elif (contentType == 'image' and usecaseType == 'deep' and custom == 'cnn'):
                # change needed here .. rethink of use case type as deep or classification
                #dataframe = cnn.(X_train, X_test, y_train)
                pass


            elif (contentType == 'text' and usecaseType == 'neural' and custom == 'perceptron'):
                # need user inputs here
                weights = []
                dataframe = perceptron.PerceptronSingleLayerClassifier(dataframe,weights,ticketId)

        elif(inputType=='labelled' and mlType=='deepLearning'):

            if(contentType=='text' and usecaseType == 'sentencePreciction' and custom == 'LSTM'):
                dataframe = ls.LSTMPredictor(dataframe,ticketId)
                log_data['CombinationID'] = ['C27']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['LSTM']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Long Short Term Memory based Recurrent Neural Network. The output model have been saved at the location'+str(os.getcwd())+os.sep+'LSTM_' + ticketId + '.sav']
            elif(contentType=='text' and usecaseType == 'classification' and custom == 'ann'):
                print('################### inside ann in master #######################')



                print(type(dataframe))
                dataframe = ann.annPredictor(dataframe,ticketId)
                log_data['CombinationID'] = ['C28']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['ANN']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Artificial Neural network. The output model have been saved at the location'+str(os.getcwd())+os.sep+'ann_' + ticketId + '.sav']
            elif(contentType == 'text' and usecaseType == 'classification' and custom == 'rbfn'):
                print('inside rbfn')
                X_train, X_test = LabelEncoding.LabelEncode(X_train), LabelEncoding.LabelEncode(X_test)
                X_train, X_test = ScaleTransform(X_train, scalingType), ScaleTransform(X_test, scalingType)
                dataframe = rbfn.RBFNClassifier(dataframe,X_train,y_train,X_test,y_test,numClusters,ticketId)
                log_data['CombinationID'] = ['C28']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['RBFN']
                log_data['Input'] = [trainingColumns]
                log_data['Output'] = [outputColumn]
                log_data['Info'] = [
                    'Radial Belief Function Network. The output model have been saved at the location'+str(os.getcwd())+os.sep+'rbfn_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'classification' and custom == 'torchEncoder'):
                encoding_dim = 32
                dataframe = encoder.autoencoder_keras(encoding_dim, X_train, X_test,ticketId)
                log_data['CombinationID'] = ['C29']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['AutoEncoder']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Stacked AutoEncoder. The output model have been saved at the location'+str(os.getcwd())+os.sep+'autoEncoder_' + ticketId + '.sav']

            elif (contentType == 'text' and usecaseType == 'classification' and custom == 'boltzmann'):
                training_set = X_train
                batch_size = 100
                nh,nb_users = 1000
                dataframe = bolt.train_re_bm(training_set, nh, batch_size, nb_users,ticketId)
                log_data['CombinationID'] = ['C30']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['DBM']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Restricted boltzmann machine. The output model have been saved at the location'+str(os.getcwd())+os.sep+'boltzmann_' + ticketId + '.sav']

                # rbm = re_bm.train_re_bm(training_set, nh, batch_size, nb_users)
                #
                # re_bm.test_re_bm(training_set, test_set, rbm, nb_users)

            elif (contentType == 'text' and usecaseType == 'classification' and custom == 'som'):

                soM, _ = som.self_organised_maps(X_train,ticketId)
                # plot som
                som.plot_som(soM, X_train, y_train)
                log_data['CombinationID'] = ['C31']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['SOM']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Self Organized Map. The output model have been saved at the location'+str(os.getcwd())+os.sep+'som_' + ticketId + '.sav']
            elif (contentType == 'image' and usecaseType == 'classification' and custom == 'cnn'):

                CNNImage.cnnClassifier(dataframe,ticketId)
                log_data['CombinationID'] = ['C32']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['CNN']
                log_data['Input'] = [X_train.columns]
                log_data['Output'] = [y_train.columns]
                log_data['Info'] = [
                    'Convolutional Nueral Network. The output model have been saved at the location'+str(os.getcwd())+os.sep+'cnn_' + ticketId + '.sav']




        else:
            if (contentType == 'text' and usecaseType == 'recomendation' and encodingType == 'label' and scalingType == 'standard' and custom == 'eclat'):

                dataframe = ec.EclatRecommendationEngine(outData,transactionIdColumn, ItemsColumn,ticketId)
                log_data['CombinationID'] = ['C19']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['Eclat Recommendation']
                log_data['Input'] = [outData.columns]
                log_data['Output'] = [outData.columns]
                log_data['Info'] = [
                    'Eclat recommendation engine. The output model have been saved at the location'+str(os.getcwd())+os.sep+'Eclat_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'recomendation' and encodingType == 'label' and scalingType == 'standard' and custom == 'apriori'):
                print('inside apriori master')
                dataframe = ap.AprioriRecomendationEngine(outData,transactionIdColumn, ItemsColumn,ticketId)
                log_data['CombinationID'] = ['C20']
                log_data['StepNo'] = ['1']
                log_data['StepDescription'] = ['Apriori Recommendation']
                log_data['Input'] = [outData.columns]
                log_data['Output'] = [outData.columns]
                log_data['Info'] = [
                    'Apriori recommendation engine. The output model have been saved at the location'+str(os.getcwd())+os.sep+'Apriori_' + ticketId + '.sav']
            elif(contentType == 'text' and usecaseType == 'clustering' and encodingType == 'label' and scalingType == 'standard' and custom == 'em'):
                dataframe = LabelEncoding.LabelEncode(dataframe)
                dataframe = em.GuassianMixturePredictor(outData,dataframe,numClusters,ticketId)
                log_data['CombinationID'] = ['C33'] * 2
                log_data['StepNo'] = list(range(1, 3))
                log_data['StepDescription'] = ['Encoding', 'Expectation Maximisaton']
                log_data['Input'] = [trainingColumns] * 2
                log_data['Output'] = [outputColumn] * 2
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Expectation Maximisation. The output model have been saved at the location'+str(os.getcwd())+os.sep+'em_' + ticketId + '.sav']
            elif(contentType == 'text' and usecaseType == 'clustering' and encodingType == 'label'  and custom == 'hierarchical'):
                print('inside hierarchical')
                dataframe = LabelEncoding.LabelEncode(dataframe)
                dataframe = hierarchical.hierarchicalClusterPredictor(outData,dataframe,numClusters,ticketId)
                log_data['CombinationID'] = ['C34'] * 2
                log_data['StepNo'] = list(range(1, 3))
                log_data['StepDescription'] = ['Encoding', 'Hierarchical Clustering']
                log_data['Input'] = [trainingColumns] * 2
                log_data['Output'] = [outputColumn] * 2
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Hierarchical Clustering. The output model have been saved at the location'+str(os.getcwd())+os.sep+'hierarchical_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'clustering' and encodingType == 'label' and scalingType == 'standard' and custom == 'dbscan'):
                print('inside db scan')
                dataframe = LabelEncoding.LabelEncode(dataframe)
                dataframe = db.dbScanClusterizer(outData,dataframe,ticketId,epsilon=epsilon,minSamples=minSamples)
                log_data['CombinationID'] = ['C35'] * 2
                log_data['StepNo'] = list(range(1, 3))
                log_data['StepDescription'] = ['Encoding', 'DBScan Clustering']
                log_data['Input'] = [trainingColumns] * 2
                log_data['Output'] = [outputColumn] * 2
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Density based Scan clustering. The output model have been saved at the location'+str(os.getcwd())+os.sep+'dbScan_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'clustering' and encodingType == 'label' and scalingType == 'standard' and custom == 'kmeans'):
                print('inside kmeans in master')
                dataframe = LabelEncoding.LabelEncode(dataframe)
                print('encoding completed')
                print('after encoding :',dataframe)
                outData = km.kmeansCluster(outData,dataframe,numClusters,ticketId)
                log_data['CombinationID'] = ['C36'] * 2
                log_data['StepNo'] = list(range(1, 3))
                log_data['StepDescription'] = ['Encoding', 'kmeans clustering']
                log_data['Input'] = [trainingColumns] * 2
                log_data['Output'] = [outputColumn] * 2
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'kmeans clustering. The output model have been saved at the location'+str(os.getcwd())+os.sep+'kmeans_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'clustering' and encodingType == 'label' and scalingType == 'standard' and custom == 'kmedians'):
                dataframe = LabelEncoding.LabelEncode(dataframe)
                dataframe = kmed.kMediansClusterPredictor(outData,dataframe,numClusters,ticketId)
                log_data['CombinationID'] = ['C37'] * 2
                log_data['StepNo'] = list(range(1, 3))
                log_data['StepDescription'] = ['Encoding', 'kmedians clustering']
                log_data['Input'] = [trainingColumns] * 2
                log_data['Output'] = [outputColumn] * 2
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'kmedians clustering. The output model have been saved at the location'+str(os.getcwd())+os.sep+'kMedians_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'clustering' and encodingType == 'label' and scalingType == 'standard' and custom == 'spectral'):
                dataframe = LabelEncoding.LabelEncode(dataframe)
                dataframe = spec.spectralCluster(outData,dataframe,numClusters,ticketId)
                log_data['CombinationID'] = ['C38'] * 2
                log_data['StepNo'] = list(range(1, 3))
                log_data['StepDescription'] = ['Encoding', 'Spectral clustering']
                log_data['Input'] = [trainingColumns] * 2
                log_data['Output'] = [outputColumn] * 2
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Spectral clustering. The output model have been saved at the location'+str(os.getcwd())+os.sep+'spectral_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'clustering' and encodingType == 'label' and scalingType == 'standard' and custom == 'affinity'):
                dataframe = LabelEncoding.LabelEncode(dataframe)
                dataframe = aff.affinityCluster(outData,dataframe,numClusters,ticketId)
                log_data['CombinationID'] = ['C39'] * 2
                log_data['StepNo'] = list(range(1, 3))
                log_data['StepDescription'] = ['Encoding', 'Affinity Propagation']
                log_data['Input'] = [trainingColumns] * 2
                log_data['Output'] = [outputColumn] * 2
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'Affinity Propagation. The output model have been saved at the location'+str(os.getcwd())+os.sep+'affinity_' + ticketId + '.sav']
            elif (contentType == 'text' and usecaseType == 'clustering' and encodingType == 'label' and scalingType == 'standard' and custom == 'meanShift'):
                dataframe = LabelEncoding.LabelEncode(dataframe)
                dataframe = mean.meanShiftCluster(outData,dataframe,bandwidth,ticketId)
                log_data['CombinationID'] = ['C40'] * 2
                log_data['StepNo'] = list(range(1, 3))
                log_data['StepDescription'] = ['Encoding', 'meanShift clustering']
                log_data['Input'] = [trainingColumns] * 2
                log_data['Output'] = [outputColumn] * 2
                log_data['Info'] = [
                    'Converting the labels into numeric form so as to convert it into the machine-readable form',
                    'meanShift clustering. The output model have been saved at the location'+str(os.getcwd())+os.sep+'meanShift_' + ticketId + '.sav']

        if(mlType=='customDeepLearning' and custom=='ageDetect'):
            ageDetect.ageGroupDetect()
        elif (mlType == 'customDeepLearning' and custom == 'childDetect'):
            childAdult.childAdultTrain()
        elif (mlType == 'customDeepLearning' and custom == 'genderDetect'):
            genderDetect.genderDetectTrain()
        elif (mlType == 'customDeepLearning' and custom == 'raceDetect'):
            raceDetect.raceDetectionTrain()
        elif (mlType == 'customDeepLearning' and custom == 'emotionDetect'):
            emotionDetect.emotionDetectorTrain()
        elif (mlType == 'customDeepLearning' and custom == 'motionDetect'):
            motionDetect.motionDetector()
        if(usecaseType == 'recomendation'):
            outData = dataframe
        elif(mlType != 'unsupervised'):
            outData['predicted'] = dataframe['predicted']
        dataframe = outData
        print(outData.head(5))
        try:
            y_pred = list(dataframe['predicted'])

            cm = confusion_matrix(y_test, y_pred)
            accuracy = (accuracy_score(y_test, y_pred))*100
            log.info('Accuracy :' + str(accuracy) + '%')
            log.info(' Confusion Matrix : '+str(cm))



            classificationReport = classification_report(y_test, y_pred)
            log.info('Classification Report :')
            log.info(classificationReport)



        except Exception as e:
            if (y_test == 'null'):
                log.info('test dataframe doesnt have output column to compare')
            else:
                log.info('The output dataframe doesnt have a predicted column')
            print(e)

    except Exception as e:
        log.info("Master Program Failed to execute")
        log.error("Error and Type \n{},\n{}".format(e, type(e)))
        log.exception(str((traceback.format_exc())))

    log_data.to_excel(r'D:\Personal\SmartIT\data\walkthrough\predicted\log_'+ticketId+'.xlsx')
    try:
        acc = accuracy
    except:
        acc = None
    return  dataframe,acc

