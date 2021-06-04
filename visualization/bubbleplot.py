import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from visualization import dataEncoding as dataEncoding
def bubbelplot(dataframe,outputPath,x,y,style, markers,ticketId=''):

    dataframe = dataEncoding.dataEncoder(dataframe)


    ax = sns.scatterplot(x, y, style,
                         markers,
                         data=dataframe)
    plt.savefig(r'' + 'ticketId' + r'.png')
if __name__ == '__main__':
    tips = sns.load_dataset("tips")
    x = "total_bill"
    y = "tip"
    style = "time"
    ticketId = '4567'
    markers = {"Lunch": "s", "Dinner": "X"}
    bubbelplot(tips, x , y, style, markers, ticketId)

