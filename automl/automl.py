    # Simple Linear Regression
# Importing the libraries
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import time
from sklearn.cluster import KMeans
import statsmodels.regression.linear_model as sm
# from pyspark import SparkContext
# from pyspark.sql import SQLContext
# sc = SparkContext.getOrCreate()

class  reg():
    global configdf
    configdf = pd.read_excel('config.xlsx', sheet_name='CONFIG')
    def trainmodel(self,filepath,independentlist,dependentlist,size,randomstate,regressionswitch,classificationswitch, pswitch, optimization, SL, CrossValidation):
        def backwardEliminationwithadjustedrsquare(x, SL):
            numVars = len(x[0])
            temp = np.zeros(X.shape).astype(int)
            for i in range(0, numVars):
                regressor_OLS = sm.OLS(y, x).fit()
                maxVar = max(regressor_OLS.pvalues).astype(float)
                adjR_before = regressor_OLS.rsquared_adj.astype(float)
                if maxVar > SL:
                    for j in range(0, numVars - i):
                        if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                            temp[:, j] = x[:, j]
                            x = np.delete(x, j, 1)
                            tmp_regressor = sm.OLS(y, x).fit()
                            adjR_after = tmp_regressor.rsquared_adj.astype(float)
                            if (adjR_before >= adjR_after):
                                x_rollback = np.hstack((x, temp[:, [0, j]]))
                                x_rollback = np.delete(x_rollback, j, 1)
                                return x_rollback
                            else:
                                continue
            regressor_OLS.summary()
            return x
        if pswitch == 'spark':
            pass
            # sqlContext = SQLContext(sc)
            # df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(filepath)
            # dataset = df.toPandas()
        else:
            dataset = pd.read_csv(filepath)
        dataset = dataset.dropna()
        X = dataset[independentlist].values
        y = dataset[dependentlist].values
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = randomstate)
        # Fitting Simple Linear Regression to the Training set
        if regressionswitch == 0:
            regressor = LinearRegression()
            if optimization == 1:
                X_opt = X[:, list(range(X.shape[1] - 1))]
                X_Modeledwithadjustedrsquare = backwardEliminationwithadjustedrsquare(X_opt, SL)
                X_linopt_train, X_linopt_test, y_linopt_train, y_linopt_test = train_test_split(X_Modeledwithadjustedrsquare, y, test_size=size, random_state=randomstate)
                regressor.fit(X_linopt_train, y_linopt_train)
            else:
                regressor.fit(X_train, y_train)

        if regressionswitch == 1:
            iter = configdf['var'][1].astype(int)
            regressor = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                           intercept_scaling=1, max_iter=iter, multi_class='ovr', n_jobs=1,
                                           penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
                                           verbose=0, warm_start=False)
            if optimization == 1:
                kfold = KFold(n_splits=3, random_state=7)
                result = cross_val_score(regressor, X, y, cv=kfold, scoring='accuracy')
                if CrossValidation == 'Grid':
                    dual = [True, False]
                    max_iter = [100, 110, 120, 130, 140]
                    param_grid = dict(dual=dual, max_iter=max_iter)
                    lr = LogisticRegression(penalty='l2')
                    regressor = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, n_jobs=-1)
                    start_time = time.time()
                    grid_result = regressor.fit(X, y)
                    # Summarize results
                    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                    print("Execution time: " + str((time.time() - start_time)) + ' ms')
                if CrossValidation == "Random":
                    random = RandomizedSearchCV(estimator=lr, param_distributions=param_grid, cv=3, n_jobs=-1)
                    start_time = time.time()
                    random_result = random.fit(X, y)
                    # Summarize results
                    print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
                    print("Execution time: " + str((time.time() - start_time)) + ' ms')

            else:
                regressor.fit(X_train, y_train)
        if regressionswitch == 2:
            dgree = configdf['var'][2].astype(int)
            poly_reg = PolynomialFeatures(degree=dgree)
            X_poly = poly_reg.fit_transform(X)
            regressor = LinearRegression()
            regressor.fit(X_poly, y)
        if regressionswitch == 3:
            alfa = configdf['var'][3].astype(int)
            regressor = linear_model.Lasso(alpha=alfa)
            regressor.fit(X_train, y_train)
        #For ridge regression, we introduce GridSearchCV. This will allow us to automatically perform 5-fold cross-validation with a range of different regularization parameters in order to find the optimal value of alpha.    
        if regressionswitch == 4:
            iter = configdf['var'][4].astype(int)
            Method = configdf['Method'][4]
            ridge = Ridge()
            parameters = {'alpha' : [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 , 10, 20]}
            regressor = GridSearchCV(ridge, parameters, scoring=Method, cv = iter)
            regressor.fit(X_train, y_train)
            print(regressor.best_params_)
            print(regressor.best_score_)
        if regressionswitch == 5:
            iter = configdf['var'][4].astype(int)
            Method = configdf['Method'][4]
            lass = Lasso()
            parameters = {'alpha' : [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 , 10, 20]}
            regressor = GridSearchCV(lass, parameters, scoring=Method, cv = iter)
            regressor.fit(X_train, y_train)
            print(regressor.best_params_)
            print(regressor.best_score_)

        if classificationswitch == 0:
            clasi = GradientBoostingClassifier()
            clasi.fit(X_train, y_train)
        if classificationswitch == 1:
            if classificationswitch == configdf['Switch'][6].astype(int):
                iter = configdf['var'][6].astype(int)
                Method = configdf['Method'][6]
            clasi = DecisionTreeClassifier(criterion=Method, max_depth=iter)
            clasi.fit(X_train, y_train)
        if classificationswitch == 2:
            if classificationswitch == configdf['Switch'][7].astype(int):
                iter = configdf['var'][7].astype(int)
                Method = configdf['Method'][7]
            clasi = RandomForestClassifier(n_estimators=iter)
            clasi.fit(X_train, y_train)
        if classificationswitch == 3:
            clasi = KNeighborsClassifier()
            clasi.fit(X_train, y_train)
        if classificationswitch == 4:
            # Create a Gaussian Classifier
            clasi = GaussianNB()
            clasi.fit(X_train, y_train)

        # saves the data in pickle file for testing purpose
        if regressionswitch != -1:
            pickle.dump(regressor, open(targetfilepath, 'wb'))
            # Predicting the Test set results
            loaded_model = pickle.load(open(targetfilepath, 'rb'))
            # model accuracy
            if optimization == 1:
                result = loaded_model.score(X_linopt_test, y_linopt_test)
            else:
                result = loaded_model.score(X_test, y_test)
            print(result,'result')
        if classificationswitch != -1:
            pickle.dump(clasi, open(targetfilepath, 'wb'))
            # Predicting the Test set results
            loaded_model = pickle.load(open(targetfilepath, 'rb'))
            # model accuracy
            result = loaded_model.score(X_test, y_test)
            print(result,'result')


    def trainunsup(self, filepath, pswitch):
        if pswitch == 'spark':
            pass
            # sqlContext = SQLContext(sc)
            # df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
            #     filepath)
            # dataset = df.toPandas()
        else:
            dataset = pd.read_csv(filepath)

        dataset = dataset.dropna()
        X = dataset[independentlist].values
        # since no dependent variable for unsupervised learning
        # y = dataset[dependentlist].values
        kmeans = KMeans(n_clusters=3,max_iter=600, algorithm = 'auto').fit(dataset)
        centroids = kmeans.cluster_centers_
        print(centroids)

    def testmodel(self, val,targetfilepath):
        loaded_model = pickle.load(open(targetfilepath, 'rb'))
        predictor = loaded_model.predict(np.asarray(val))[0]
        print(predictor)

    # def visualization(self):
    #      # Visualising the Training set results
    #     plt.scatter(X_train, y_train, color = 'red')
    #     plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    #     plt.title('Salary vs Experience (Training set)')
    #     plt.xlabel('Years of Experience')
    #     plt.ylabel('Salary')
    #     plt.show()
    #     #
    #      # Visualising the Test set results
    #     plt.scatter(X_test, y_test, color = 'red')
    #     plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    #     plt.title('Salary vs Experience (Test set)')
    #     plt.xlabel('Years of Experience')
    #     plt.ylabel('Salary')
    #     plt.show()
if __name__ == '__main__':
    clsobj = reg()
    targetfilepath= "model.sav"
    # val = [[10.8   ,  0.47  ,  0.43  ,  2.1   ,  0.171 , 27.    , 66.    , 0.9982,  3.17  ,  0.76  , 10.8   ]]
    # independentlist = ['Existing_EMI', 'Loan_Amount_Applied', 'Loan_Tenure_Applied', 'Monthly_Income', 'Var4', 'Var5', 'Age', 'EMI_Loan_Submitted_Missing', 'Interest_Rate_Missing', 'Loan_Amount_Submitted_Missing', 'Loan_Tenure_Submitted_Missing', 'Processing_Fee_Missing', 'Device_Type_0', 'Device_Type_1', 'Filled_Form_0', 'Filled_Form_1', 'Gender_0', 'Gender_1', 'Var1_0', 'Var1_1', 'Var1_2', 'Var1_3', 'Var1_4', 'Var1_5', 'Var1_6', 'Var1_7', 'Var1_8', 'Var1_9', 'Var1_10', 'Var1_11', 'Var1_12', 'Var1_13', 'Var1_14', 'Var1_15', 'Var1_16', 'Var1_17', 'Var1_18', 'Var2_0', 'Var2_1', 'Var2_2', 'Var2_3', 'Var2_4', 'Var2_5', 'Var2_6', 'Mobile_Verified_0', 'Mobile_Verified_1', 'Source_0', 'Source_1', 'Source_2']
    independentlist = ['R&D Spend','Administration','Marketing Spend']
    dependentlist = ['Profit']
    size = 1/3
    randomstate = 0
    filepath='50_Startups.csv'
    regressionswitch = 0
    classificationswitch = -1
    supervised = 0
    pswitch = 'python'
    optimization = 1
    SL = 0.5
    CrossValidation = 'Random'
    if supervised == 0:
        clsobj.trainmodel(filepath, independentlist, dependentlist, size, randomstate, regressionswitch, classificationswitch, pswitch, optimization, SL, CrossValidation)
    if supervised == 1:
        clsobj.trainunsup(filepath, pswitch)
# =============================================================================
#     clsobj.testmodel(val,targetfilepath)
# =============================================================================
# =============================================================================
#     clsobj.visualization()
# =============================================================================
