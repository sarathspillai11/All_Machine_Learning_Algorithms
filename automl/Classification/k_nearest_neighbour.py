from sklearn.neighbors import KNeighborsClassifier

def knn(x_train, y_train, x_test):
    clasi = KNeighborsClassifier()
    clasi.fit(x_train, y_train)
    out = clasi.predict(x_test)
    return out