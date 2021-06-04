# Use MinMaxScaler as your default
# Use RobustScaler if you have outliers and can handle a larger range
# Use StandardScaler if you need normalized features
# Use Normalizer sparingly - it normalizes rows, not columns

import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
class switch(object):
    value = None
    def __new__(class_, value):
        class_.value = value
        return True

def case(*args):
    return any((arg == switch.value for arg in args))
def Datapreprocessing(x_train, pswitch,norm='l1'):
    # create list of column names to use later
    col_names = list(x_train.columns)
    while switch(pswitch):
        if case('RobustScaler'):
            # RobustScaler subtracts the column median and divides by the interquartile range.
            r_scaler = preprocessing.RobustScaler()
            x_train = r_scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=col_names)
            return x_train
            break
        if case('StandardScaler'):
            # StandardScaler is scales each column to have 0 mean and unit variance.
            s_scaler = preprocessing.StandardScaler()
            x_train = s_scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=col_names)
            return x_train
            break
        if case('MinMaxScalar'):
            # StandardScaler is scales each column to have 0 mean and unit variance.
            s_scaler = preprocessing.MinMaxScaler()
            x_train = s_scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=col_names)
            return x_train
            break
        if case('MaxAbsScaler'):
            # StandardScaler is scales each column to have 0 mean and unit variance.
            s_scaler = preprocessing.MinMaxScaler()
            x_train = s_scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=col_names)
            return x_train
            break
        if case('Normalizer'):
            # Note that normalizer operates on the rows, not the columns. It applies l2 normalization by default.
            n_scaler = preprocessing.Normalizer(norm)
            x_train = n_scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train, columns=col_names)
            return x_train
            break


if __name__ == '__main__':
    train = pd.read_csv(r'C:\machinlearning-trainings-master\allregressions\decisiontree.csv')
    pswitch = 'Normalizer'
    norm='l1'
    d = Datapreprocessing(train,pswitch,norm)


