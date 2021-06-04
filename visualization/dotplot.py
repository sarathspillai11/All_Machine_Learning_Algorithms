import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from visualization import dataEncoding as dataEncoding
def dotPlotter(dataframe,outputPath,x,y,ticketId=''):
    dataframe = dataEncoding.dataEncoder(dataframe)
    sns.scatterplot(x, y, palette="RdYlGn",
                         data=dataframe)
    plt.savefig(r'' + ticketId + r'.png')
if __name__ == '__main__':
    tips = sns.load_dataset("tips")
    x = "total_bill"
    y = "tip"
    ticketId = '4567'
    dotPlotter(tips, x , y, ticketId)

