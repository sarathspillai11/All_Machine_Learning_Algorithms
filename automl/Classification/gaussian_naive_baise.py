from sklearn.naive_bayes import GaussianNB
def gaussian_naive(x_train, y_train, x_test):
    clasi = GaussianNB()
    clasi.fit(x_train, y_train)
    out = clasi.predict(x_test)
    return out