import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def clusterMapper(dataframe,outputPath,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    data_Len = (dataframe.shape)[0]
    print('number of rows : ',data_Len)
    #print('current indexes : ',list(dataframe.index))
    #dataframe.set_index(range(data_Len))
    #dataframe = dataframe.pivot(x,y,z)
    g = sns.clustermap(dataframe,metric="correlation")
    plt.savefig(outputPath+r'\clusterMap_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    clusterMapper(dataframe=data, ticketId='4567')

