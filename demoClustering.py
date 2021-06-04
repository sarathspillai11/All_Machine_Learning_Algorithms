import pandas as pd
import Master.MasterAlgorithm as master
#from Master import masterVisualisation as visualise
from sklearn.model_selection import train_test_split
import numpy as np


data = pd.read_excel(r"D:\Personal\SmartIT\data\Credit Card Fraud\train.xlsx")


'''
kmeans_out =  master.findCombination(ticketId='23082020'
                             ,dataframe=data,X_train=data,numClusters=2,inputType = 'unlabelled',mlType='unsupervised',contentType = 'text',usecaseType = 'clustering',encodingType = 'label',custom = 'kmeans')
(kmeans_out[0]).to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_kmeans_23082020.xlsx')


dbscan_out = master.findCombination(ticketId='123'
                                  , dataframe=data, X_train=data,epsilon=0.01,minSamples=2,inputType = 'unlabelled',mlType='unsupervised',contentType = 'text' , usecaseType = 'clustering' , encodingType = 'label' , scalingType = 'standard' , custom = 'dbscan'  )
(dbscan_out[0]).to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_dbscan_23082020.xlsx')'''




ann_out = master.findCombination(dataframe=data, inputType='labelled', mlType='deepLearning',usecaseType = 'classification',
                                 contentType='text', custom='ann',ticketId='23082020')
(ann_out[0]).to_excel(r'D:\Personal\SmartIT\data\walkthrough\demo\predicted\Predicted_ann_23082020.xlsx')


