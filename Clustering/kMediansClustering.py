import pandas as pd
from sklearn.preprocessing import normalize
from Encoding import LabelEncoding as le
from Scaling import StandardScaling as Sc
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from savemodel import saveas_sav


def kMediansClusterPredictor(fullData,dataframe,numClusters,ticketId):

    data_manhattan = pairwise_distances(dataframe,metric='manhattan')
    c = SpectralClustering(n_clusters=numClusters, affinity='precomputed', assign_labels="discretize",random_state=0)
    c.fit(data_manhattan)
    clusters = list(c.labels_)
    fullData['predicted'] = clusters
    saveas_sav(c, 'kMedians_' + ticketId + '.sav')
    return fullData