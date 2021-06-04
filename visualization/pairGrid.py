import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def pairGridPlotter(dataframe,outputPath,hue,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    data_Len = (dataframe.shape)[0]
    print('number of rows : ',data_Len)
    #print('current indexes : ',list(dataframe.index))
    #dataframe.set_index(range(data_Len))
    #dataframe = dataframe.pivot(x,y,z)
    g = sns.PairGrid(dataframe,hue=hue)
    g = g.map(plt.scatter)

    g = g.add_legend()
    plt.savefig(outputPath+r'\pairGrid_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_excel(r"D:\Personal\SmartIT\data\diabetes.xlsx")
    pairGridPlotter(dataframe=data, ticketId='4567',hue='AgeCategory')

