from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
from savemodel import saveas_sav
from sklearn.feature_extraction.text import CountVectorizer


def multinomialNaiveClassifier(dataframe,x_train, y_train, x_test,ticketId):

    print('inside naive original module')


    # myList = np.arange(0.00001, 0.001, 0.00005)
    # # empty list that will hold cv scores
    # cv_scores = []
    #
    # # split the train data set into cross validation train and cross validation test
    # X_tr, X_cv, y_tr, y_cv = train_test_split(x_train, y_train, test_size=0.3)
    #
    # for i in myList:
    #     nb = MultinomialNB(alpha=i)
    #     model = nb.fit(X_tr, y_tr)
    #
    #     # predict the response on the crossvalidation train
    #     pred = model.predict(X_cv)
    #
    #     # evaluate CV accuracy
    #     acc = accuracy_score(y_cv, pred, normalize=True)
    #     cv_scores.append(acc)
    #
    # # changing to misclassification error
    # MSE = [1 - x for x in cv_scores]

    # determining best alpha
    # optimal_alpha = myList[MSE.index(min(MSE))]
    optimal_alpha = 0.001
    print('\nThe optimal alpha is ', optimal_alpha)

    # vect = CountVectorizer()
    # vect.fit(x_train)
    # x_train = vect.fit_transform(x_train)

    nb = MultinomialNB(alpha=optimal_alpha)
    model = nb.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    dataframe['predicted'] = y_pred
    saveas_sav(model, 'naiveBayes_' + ticketId + '.sav')
    return dataframe