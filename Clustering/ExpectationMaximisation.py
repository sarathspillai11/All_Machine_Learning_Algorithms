import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import numpy as np
from savemodel import saveas_sav
def GuassianMixturePredictor(dataframe, x_train, numClusters,ticketId):

    gmm = GMM(n_components=numClusters).fit(x_train)
    labels = gmm.predict(x_train)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=labels, s=40, cmap='viridis')
    dataframe['predicted'] = labels
    saveas_sav(gmm, 'em_' + ticketId + '.sav')
    return dataframe

# if __name__ == '__main__':
#     df = pd.read_excel(r'C:\Users\KPMG\ELMER\QRM2017_2019_new.xlsx')
#     df.replace(r'\s+', np.nan, regex=True)
#     x_train = df.iloc[:, :-1].values
#     y_train = df.iloc[:, -1].values
#     GuassianMixturePredictor(x_train, y_train)
