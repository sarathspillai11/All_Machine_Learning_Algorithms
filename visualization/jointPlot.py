import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def jointPlotter(dataframe,outputPath,x,y,type='hex',ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.jointplot(x=x, y=y, data=dataframe, kind=type)
    plt.savefig(outputPath+r'\jointPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    jointPlotter(dataframe=data, ticketId='4567',x="GarageArea", y="SalePrice",type='reg')

