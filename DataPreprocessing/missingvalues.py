import numpy as np
def replacemissingvalue(train):
    findnan = train.isnull().sum()
    nanindices = [i for i, x in enumerate(findnan) if x >0]
    if len(nanindices) > 0:
        for i in nanindices:
            #print(train[train.columns.tolist()[i]].dtype.name)
            if train[train.columns.tolist()[i]].dtype.name == 'category':
                train[train.columns.tolist()[i]] = train[train.columns.tolist()[i]].replace(np.nan, train[train.columns.tolist()[i]].mode())
            else:
                train[train.columns.tolist()[i]] = train[train.columns.tolist()[i]].replace(np.nan, train[train.columns.tolist()[i]].mode())
    return train                          

# if __name__ == '__main__':
#     import pandas as pd
#     train = pd.read_csv(r'C:\machinlearning-trainings-master\allregressions\decisiontree.csv')
#     d = replacemissingvalue(train)
