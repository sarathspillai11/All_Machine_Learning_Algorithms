# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:09:20 2020

@author: vamsi
"""

#https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
def hierarchical_clustering(data_scaled,n_clusters=2, affinity='euclidean', linkage='ward'):
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    plt.axhline(y=6, color='r', linestyle='--')
    cluster = AgglomerativeClustering(n_clusters, affinity, linkage)
    cluster.fit_predict(data_scaled)
    plt.figure(figsize=(10, 7))
    plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_)




if __name__ == '__main__':
    filename = 'Wholesale customers data.csv'
    data = pd.read_csv(filename)
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    hierarchical_clustering(data_scaled,n_clusters=2, affinity='euclidean', linkage='ward')