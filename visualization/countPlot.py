import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def countPlotter(dataframe,outputPath,ticketId='',category=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    ax = sns.countplot(x=category, data=dataframe)
    plt.savefig(outputPath+r'\CountPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_excel(r"D:\Personal\SmartIT\data\diabetes.xlsx")
    countPlotter(data,ticketId='4567',category='AgeCategory')
