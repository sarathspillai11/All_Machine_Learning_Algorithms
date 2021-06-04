import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def heatMapper(dataframe,outputPath,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    data_Len = (dataframe.shape)[0]
    print('number of rows : ',data_Len)
    #print('current indexes : ',list(dataframe.index))
    #dataframe.set_index(range(data_Len))
    #dataframe = dataframe.pivot(x,y,z)
    g = sns.heatmap(dataframe)
    plt.savefig(outputPath+r'\heatMap_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    heatMapper(dataframe=data, ticketId='4567')

