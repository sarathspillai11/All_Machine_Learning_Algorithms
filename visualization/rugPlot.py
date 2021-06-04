import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def rugPlotter(dataframe,outputPath,x,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.rugplot(dataframe[x],axis='x')
    plt.savefig(outputPath+r'\rugPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    rugPlotter(dataframe=data, ticketId='4567',x="GarageArea",)

