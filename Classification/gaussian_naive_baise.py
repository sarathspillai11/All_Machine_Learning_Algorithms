from sklearn.naive_bayes import GaussianNB
from savemodel import saveas_sav
def gaussian_naive(dataframe,x_train, y_train, x_test,ticketId,numClasses):
    clasi = GaussianNB()
    clasi.fit(x_train, y_train)
    out = clasi.predict(x_test)
    dataframe['predicted'] = out
    saveas_sav(clasi, 'guassian_naive_' + ticketId + '.sav')
    return dataframe