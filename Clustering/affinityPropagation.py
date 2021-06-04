from sklearn.cluster import AffinityPropagation
from savemodel import saveas_sav

def affinityCluster(fulldata,dataset,numClusters,ticketId):
    #kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, algorithm=algorithm).fit(dataset)
    clustering = AffinityPropagation().fit(dataset)
    centroids = clustering.labels_
    fulldata['predicted'] = centroids
    saveas_sav(clustering, 'affinity_' + ticketId + '.sav')
    return fulldata