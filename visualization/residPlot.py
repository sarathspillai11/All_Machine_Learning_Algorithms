import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def residPlotter(dataframe,outputPath,x,y,lowess,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    g = sns.residplot(x=x, y=y, data=dataframe,lowess=lowess)
    plt.savefig(outputPath+r'\residPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    residPlotter(dataframe=data, ticketId='4567',x="GarageArea", y="SalePrice",lowess=True)

