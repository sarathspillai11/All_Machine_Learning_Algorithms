from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.preprocessing import StandardScaler
from savemodel import saveas_sav
def dbScanClusterizer(fullDataframe,dataframe,ticketId,epsilon=0.3,minSamples=10):

    """ DBSCAN object that requires a minimum of minSamples number of data points in a neighborhood of radius epsilon to be considered a core point.

    """
    # dataframe = StandardScaler.fit_transform(dataframe)
    print('inside db scan : ',epsilon,minSamples)
    db = DBSCAN(eps=epsilon,min_samples=minSamples)
    db.fit(dataframe)
    cluster_labels = list(db.labels_)
    # dataframe = pd.DataFrame(dataframe)
    fullDataframe['cluster'] = cluster_labels
    print('cluster labels :',set(cluster_labels))
    saveas_sav(db, 'dbScan_' + ticketId + '.sav')
    return fullDataframe
