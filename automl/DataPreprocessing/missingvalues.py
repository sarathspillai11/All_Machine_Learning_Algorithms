def replacemissingvalue(train):
    findnan = train.isnull().sum()
    nanindices = [i for i, x in enumerate(findnan) if x >0]
    if len(nanindices) > 0:
        for i in nanindices:
            print(train[train.columns.tolist()[i]]) 


if __name__ == '__main__':
    import pandas as pd
    train = pd.read_csv(r'C:\machinlearning-trainings-master\allregressions\decisiontree.csv')
    d = replacemissingvalue(train)
