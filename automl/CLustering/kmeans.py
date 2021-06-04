from sklearn.cluster import KMeans

def gradient_boost_classfier(dataset, n_clusters, max_iter, algorithm):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, algorithm=algorithm).fit(dataset)
    centroids = kmeans.cluster_centers_
    return centroids


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('50_Startups.csv')
    n_clusters = 3
    max_iter = 600
    algorithm = 'auto'
    gradient_boost_classfier(dataset, n_clusters, max_iter, algorithm)