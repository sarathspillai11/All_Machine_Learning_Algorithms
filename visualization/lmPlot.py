import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def lmPlotter(dataframe,outputPath,x,y,hue,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.lmplot(x=x, y=y, hue=hue, data=dataframe)
    plt.savefig(outputPath+r'\lmPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    lmPlotter(dataframe=data, ticketId='456',x="GarageArea", y="SalePrice", hue="Street")

