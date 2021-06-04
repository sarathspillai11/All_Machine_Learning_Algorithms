import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer

def logistic_regression(Xtrain, Ytrain, Xtest, Ytest, trainedinput, trainedoutput, BATCH_SIZE, TEST_BATCH_SIZE, N_EPOCHS):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(trainedinput, trainedoutput)

        def forward(self, x):
            x = self.fc1(x)
            return F.log_softmax(x, dim=-1)


    model = Net()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.NLLLoss()


    Xtrain_ = Variable(torch.from_numpy(np.asarray(Xtrain)))
    Xtest_ = Variable(torch.from_numpy(np.asarray(Xtest)))
    Ytrain_ = Variable(torch.from_numpy(np.asarray(Ytrain).astype(np.int64)))
    Ytest_ = Variable(torch.from_numpy(np.asarray(Ytest).astype(np.int64)))


    perfs = []


    for t in range(1, N_EPOCHS + 1):
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Forward pass: compute predicted y by passing x to the model.
        Ypred = model(Xtrain_)

        # Compute and print loss.
        loss = loss_fn(Ypred , Ytrain_)

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        Ypred_test = model(Xtest_)
        loss_test = loss_fn(Ypred_test, Ytest_)
        pred = Ypred_test.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        accuracy = pred.eq(Ytest_.data.view_as(pred)).cpu().sum().item() / Ytest.size
        perfs.append([t, loss.data.item(), loss_test.data.item(), accuracy])

    print(pred)


    df_perfs = pd.DataFrame(perfs, columns=["epoch", "train_loss", "test_loss", "accuracy"]).set_index("epoch")
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    print("Last accuracy %.3f" % df_perfs.accuracy.iloc[-1])
    df_perfs[["train_loss", "test_loss"]].plot(ax=ax1);
    df_perfs[["accuracy"]].plot(ax=ax2);
    plt.ylim(ymin=0.7);

if __name__ == '__main__':
# =============================================================================
#     X, Y = load_iris(return_X_y=True)
#     # mandatory to convert independent values to float32 as optimizer is not able to accept double.
#     X = X.astype("float32")
#     Y = Y.astype("float32")
# =============================================================================
    
    D = pd.read_csv(r"C:\Users\vamsi\PycharmProjects\auto_ml_pytorch\datasets\iris.csv",
                                 header=None)
    

    # We extract all rows and the first 2 columns, and then transpose it
    X = D.iloc[:, 0:4].values

    # We extract all rows and the last column, and transpose it
    Y = D.iloc[:, 4] 
    
    # mandatory to convert independent values to float32 as optimizer is not able to accept double.
    X = X.astype("float32")
    Y = Y.astype("float32")

    ftrain = np.arange(X.shape[0]) % 4 != 0
    Xtrain, Ytrain = X[ftrain, :], Y[ftrain]
    Xtest, Ytest = X[~ftrain, :], Y[~ftrain]

# here 4 is no of independent feature shape, 3 is no of dependent feature shape
    trainedinput, trainedoutput = 4, 3
    BATCH_SIZE, TEST_BATCH_SIZE, N_EPOCHS = 64, 64, 2000
    lo = logistic_regression(Xtrain, Ytrain, Xtest, Ytest, trainedinput, trainedoutput, BATCH_SIZE, TEST_BATCH_SIZE, N_EPOCHS)