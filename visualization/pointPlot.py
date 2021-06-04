import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def pointPlotter(dataframe,outputPath,x,y,hue,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.pointplot(x=x, y=y, hue=hue, data=dataframe)
    plt.savefig(outputPath+r'\pointPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    pointPlotter(dataframe=data, ticketId='4567',x="GarageArea", y="SalePrice", hue="Street")

