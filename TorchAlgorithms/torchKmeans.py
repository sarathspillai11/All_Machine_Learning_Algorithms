import torch
import numpy as np
from kmeans_pytorch import kmeans
import pandas as pd

# data
num_clusters = 3

data = pd.read_excel(r"D:\Personal\SmartIT\data\clusteringSample.xlsx")
x = data.values
x = torch.from_numpy(x)


# kmeans
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclid device=torch.device('cuda:0')
)

print('cluster ids : ',cluster_ids_x)
print('cluster centers :',cluster_centers)