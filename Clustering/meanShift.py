from sklearn.cluster import MeanShift
from savemodel import saveas_sav

def meanShiftCluster(fulldata,dataset, bandwidth,ticketId):
    #kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, algorithm=algorithm).fit(dataset)

    clustering = MeanShift(bandwidth=bandwidth).fit(dataset)
    centroids = clustering.labels_
    fulldata['predicted'] = centroids
    saveas_sav(clustering, 'meanShift_' + ticketId + '.sav')
    return fulldata


# if __name__ == '__main__':
#     import pandas as pd
#     dataset = pd.read_csv('50_Startups.csv')
#     n_clusters = 3
#     max_iter = 600
#     algorithm = 'auto'
#     gradient_boost_classfier(dataset, n_clusters, max_iter, algorithm)