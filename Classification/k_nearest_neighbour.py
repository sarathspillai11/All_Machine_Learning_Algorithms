from sklearn.neighbors import KNeighborsClassifier
from savemodel import saveas_sav

def knn(dataframe,x_train, y_train, x_test,ticketId):
    clasi = KNeighborsClassifier()
    clasi.fit(x_train, y_train)
    out = clasi.predict(x_test)
    dataframe['predicted'] = out
    saveas_sav(clasi, 'knn_' + ticketId + '.sav')
    return dataframe