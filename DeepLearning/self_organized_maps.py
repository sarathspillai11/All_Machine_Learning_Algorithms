# Self Organizing Maps

# Unsupervised Deep Learning identify patterns high dimension dataset
# one of these patterns will find the fradulant way
# segmentcorrespond to specific range of values SOM
# customers are input to new space, each neuron initialised,

# winning node, guassian neighbouring function closer to the point, input space
# output space reduces dimensions, obtain with all the winning nodes
# frauds are outliers, how to detect, mean euclidean distance between in neighborhood,
# far in neurons in self-organizing maps, inverse function input associated with winning node
from pylab import bone, pcolor, colorbar, plot, show
def self_organised_maps(X):

    # Training the SOM
    # Unsupervised Learning, we don't consider
    # sigma is the radius of the different neighborhoods
    # learning weight, hyperparameter decides how much weight
    # higher the learning rates, faster will be convergence
    # lower the learning rate, the slower it takes for SOM to build
    from minisom import MiniSom
    som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

    som.random_weights_init(X)

    som.train_random(data = X, num_iteration = 100)
    return som, som.distance_map().T

# Visualising the results
# two-dimensional grid of the winning nodes
# get M-ID Mean Inter-neruon Distances, Inside the neighborhood, radius, winning 
# higher MID, winning, outlier neuron far from the general neuron, fraud, winning nodes
# with the higher M-ID. Winning node will use different nodes

def plot_som(som, X, y):
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o', 's']
    colors = ['r', 'g']
    for i, x in enumerate(X):
        w = som.winner(x)
        plot(w[0] + 0.5,
             w[1] + 0.5,
             markers[y[i]],
             markeredgecolor = colors[y[i]],
             markerfacecolor = 'None',
             markersize = 10,
             markeredgewidth = 2)
    show()

if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('Credit_Card_Applications.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    X = sc.fit_transform(X)

    # train som
    som, _ = self_organised_maps(X)
    #plot som
    plot_som(som, X, y)