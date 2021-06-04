from sklearn.cluster import SpectralClustering
from savemodel import saveas_sav

def spectralCluster(fulldata,dataset, n_clusters,ticketId):

    spectral_model_rbf = SpectralClustering(n_clusters = n_clusters, affinity ='rbf')
    labels_rbf = spectral_model_rbf.fit_predict(dataset)
    fulldata['predicted'] = labels_rbf
    saveas_sav(labels_rbf, 'spectral_' + ticketId + '.sav')
    return fulldata


