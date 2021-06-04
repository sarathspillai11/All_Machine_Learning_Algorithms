import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def linePlotter(dataframe,outputPath,x,y,hue,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.lineplot(x=x, y=y, hue=hue, data=dataframe)
    plt.savefig(outputPath+r'\linePlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    linePlotter(dataframe=data, ticketId='4567',x="GarageArea", y="SalePrice", hue="Street")

