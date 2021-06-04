
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from savemodel import saveas_sav
def annPredictor(dataset,ticketId):
    print('inside ann')

    # # Importing the dataset
    # dataset = pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, -1].values

    # Encoding categorical data

    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]

    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state = 0)

    try:
        outData = pd.concat([X_test, y_test], axis=1)
    except:
        outData = X_test

    # Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Part 2 - Now let's make the ANN!



    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = (dataset.shape)[0] + 9))

    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)

    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)


    # Making the Confusion Matrix

    #cm = confusion_matrix(y_test, y_pred)

    #print('confusion matrix :',cm)
    # print('############################# ',outData.shape)
    # print('######################## ',len(list(y_pred)))
    # print(y_pred)
    final = []
    for pred in y_pred:
        if(pred[0]>0.5):
            final.append(1)
        else:
            final.append(0)



    outData = pd.DataFrame(outData)
    outData['predicted'] = final

    saveas_sav(classifier, 'ann_' + ticketId + '.sav')
    return outData