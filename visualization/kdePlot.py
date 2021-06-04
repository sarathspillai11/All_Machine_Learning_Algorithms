import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def kdePlotter(dataframe,outputPath,x,y,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.kdeplot(dataframe[x],dataframe[y],shade=True)
    plt.savefig(outputPath+r'\kdePlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    kdePlotter(dataframe=data, ticketId='4567',x="GarageArea",y="SalePrice")

