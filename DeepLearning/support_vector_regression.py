# HINGE LOSS
import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from savemodel import saveas_sav

torch.manual_seed(42)

def supportvectorregression(dataframe,Xtrain, Ytrain, Xtest, Ytest, trainedinput, trainedoutput,ticketId):
    class SVR(nn.Module):
        def __init__(self):
            super(SVR, self).__init__()
            self.linearModel = nn.Linear(trainedinput, trainedoutput)

        def forward(self, x):
            x = self.linearModel(x)
            return x


    model = SVR()


    def hingeLoss(outputVal, dataOutput, model):
        loss1 = torch.sum(torch.clamp(1 - torch.matmul(outputVal.t(), dataOutput), min=0))
        loss2 = torch.sum(model.linearModel.weight ** 2)  # l2 penalty
        totalLoss = loss1 + loss2
        return (totalLoss)


    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    X = np.asarray(Xtrain)
    Y = np.asarray(Ytrain)

    for epoch in range(10000):
        inputVal = Variable(torch.from_numpy(X))
        outputVal = Variable(torch.from_numpy(Y))
        optimizer.zero_grad()
        modelOutput = model(inputVal)
        totalLoss = hingeLoss(outputVal, modelOutput, model)
        totalLoss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 100, totalLoss))

    Xtest = np.asarray(Xtest)
    Xtest = Variable(torch.from_numpy(Xtest))
    pred_y = model(Xtest)
    dataframe['predicted'] = pred_y
    saveas_sav(model, 'torchSVR_' + ticketId + '.sav')
    return dataframe

