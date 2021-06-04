import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def regPlotter(dataframe,outputPath,x,y,confidence_interval=70,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.regplot(x=x, y=y, ci=confidence_interval, data=dataframe,marker='+')
    plt.savefig(outputPath+r'\regPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    regPlotter(dataframe=data, ticketId='456',x="GarageArea", y="SalePrice",confidence_interval=68)

