from sklearn.preprocessing import StandardScaler

def StandardScale(X_train,X_test):
    sc =StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train,X_test