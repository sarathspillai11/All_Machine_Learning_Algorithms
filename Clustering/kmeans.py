from sklearn.cluster import KMeans
from savemodel import saveas_sav

def kmeansCluster(dataframe,dataset, n_clusters,ticketId):
    #kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, algorithm=algorithm).fit(dataset)
    kmeans = KMeans(n_clusters=n_clusters).fit(dataset)
    cluster_labels = kmeans.labels_
    print('length of centroids :',cluster_labels)
    dataframe['predicted'] = cluster_labels
    saveas_sav(kmeans, 'kmeans_' + ticketId + '.sav')
    return dataframe


