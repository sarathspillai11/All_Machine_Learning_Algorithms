import pandas as pd
import matplotlib.pyplot as plt
import visualization.dataEncoding as dataEncoding
from sklearn.model_selection import train_test_split


def tufteBoxPlotter(dataframe,outputPath,x,ticketId=''):
    data = dataEncoding.dataEncoder(dataframe)
    labels = list(data[x])
    fs = 10
    fig, axes = plt.subplots()
    axes.boxplot(data, labels=labels, showbox=False, showcaps=False)
    tufte_title = 'Tufte Style \n(showbox=False,\nshowcaps=False)'
    axes.set_title(tufte_title, fontsize=fs)
    fig.subplots_adjust(hspace=0.4)
    # plt.show()
    plt.savefig(outputPath+r'\tufteBoxPlot_' + ticketId + r'.png')



if __name__ == '__main__':

    data = pd.read_csv(r"D:\Personal\SmartIT\data\MARS House Prices\train.csv")
    X_train, data = train_test_split(data, test_size=0.01, random_state=42)
    data = dataEncoding.dataEncoder(data)
    tufteBoxPlotter(data,'LotConfig','12345')

    #
    # def tufteBoxPlotter(dataframe,outputPath,x,y,lineWidth,ticketId=')