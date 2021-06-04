from sklearn.decomposition import PCA

def principalComponentConverter(data,noOfComponents):
    pcaData = PCA(n_components=noOfComponents)
    data = pcaData.fit_transform(data)
    return data