import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd
from numpy import median
def barPlotter(dataframe,outputPath,x,y,hue,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.barplot(x=x, y=y, hue=hue, data=dataframe,estimator=median)
    plt.savefig(outputPath+r'\barPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    barPlotter(dataframe=data, ticketId='4567',x="GarageArea", y="SalePrice", hue="Street")

