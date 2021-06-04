import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def stripPlotter(dataframe,outputPath,x,y,lineWidth,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.stripplot(x=x, y=y, lineWidth=lineWidth, data=dataframe)
    plt.savefig(outputPath+r'\stripPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    stripPlotter(dataframe=data, ticketId='4567',x="GarageArea", y="SalePrice", lineWidth=1)

