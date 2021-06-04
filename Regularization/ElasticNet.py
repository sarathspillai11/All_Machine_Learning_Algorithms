from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
import pandas as pd
from savemodel import saveas_sav
from Vectorization.CountVectorization import countVectorize
from Vectorization.tfIdfVectorization import tfIdfVectorize


def ElasticNetPredictor(dataframe,x_train,y_train,x_test,ticketId):

    regr = ElasticNet(random_state=0)
    regr.fit(x_train,y_train)
    #print(regr.predict(x_test))
    y_hat = regr.predict(x_test)
    dataframe['predicted'] = y_hat
    saveas_sav(regr, 'elasticNet_' + ticketId + '.sav')
    return dataframe

# if __name__ == '__main__':
#
#     # normal numerical values
#
#     X, y = make_regression(n_features=2, random_state=0)
#
#     x_test = [[0, 0]]
#
#     print('Prediction for numeric data :')
#     ElasticNetPredictor(X,y,x_test)
#
#     # text data
#
#
#
#     textData = pd.read_excel(r'D:\Personal\SmartIT\test files\regression input.xlsx',sheet_name='trainData')
#     testData = pd.read_excel(r'D:\Personal\SmartIT\test files\regression input.xlsx',sheet_name='testData')
#
#     print(textData['Sentence'])
#
#     x_train = countVectorize(textData)
#
#     print('X TRAIN : ',x_train)
#     y_train = textData.iloc[:,-1]
#     print('Y TRAIN : ', y_train)
#     x_test = countVectorize(testData)
#
#     print('prediction for textual data with count:')
#     ElasticNetPredictor(x_train, y_train, x_test)
#
#     x_train = tfIdfVectorize(textData)
#
#     print('X TRAIN : ', x_train)
#     y_train = textData.iloc[:, -1]
#     print('Y TRAIN : ', y_train)
#     x_test = countVectorize(testData)
#
#     print('prediction for textual data with TFIDF :')
#     ElasticNetPredictor(x_train, y_train, x_test)







