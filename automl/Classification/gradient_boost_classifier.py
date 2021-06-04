from sklearn.ensemble import GradientBoostingClassifier
def gradient_boost_classfier(x_train, y_train, x_test):
    clasi = GradientBoostingClassifier()
    clasi.fit(x_train, y_train)
    out = clasi.predict(x_test)
    return out