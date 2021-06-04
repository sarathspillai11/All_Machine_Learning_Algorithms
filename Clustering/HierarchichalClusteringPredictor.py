import pandas as pd
from sklearn.preprocessing import normalize
from Encoding import LabelEncoding as le
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from savemodel import saveas_sav

def hierarchicalClusterPredictor(dataframe,x_train,numClusters,ticketId):

    print('inside hierarchical module')
    # dataframe  = le.LabelEncode(dataframe)
    # x_train = normalize(x_train)
    # plt.figure(figsize=(10, 7))

    # plt.title("Dendrograms")

    # dend = shc.dendrogram(shc.linkage(x_train, method='ward'))

    # plt.axhline(y=6, color='r', linestyle='--')


    print('number of clusters :',type(numClusters))
    cluster = AgglomerativeClustering(n_clusters=numClusters, affinity='euclidean', linkage='ward')

    cluster.fit_predict(x_train)

    # plt.figure(figsize=(10, 7))
    # plt.show()

    #plt.scatter(dataframe['Milk'], dataframe['Grocery'], c=cluster.labels_)
    print('cluster labels are :',list(cluster.labels_))
    clustersLabel = list(cluster.labels_)
    #dataframe = pd.DataFrame(dataframe)
    dataframe['predicted'] = clustersLabel
    saveas_sav(cluster, 'hierarchical_' + ticketId + '.sav')
    return dataframe


