import pandas as pd
from Encoding import LabelEncoding

def dataEncoder(dataframe):
    dataColumns = list(dataframe.columns)
    print(list(dataframe.columns))
    dataframe = LabelEncoding.LabelEncode(dataframe.values)
    dataframe = pd.DataFrame(dataframe, columns=dataColumns,dtype='float')
    return dataframe