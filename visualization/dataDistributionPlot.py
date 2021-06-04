import matplotlib.pyplot as plt
import seaborn as sns
import visualization.dataEncoding as dataEncoding
import pandas as pd

def distributionPlotter(dataframe,outputPath,ticketId='',size=5,hue='',category=''):

    dataframe = dataEncoding.dataEncoder(dataframe)
    sns.FacetGrid(dataframe, hue=hue, size=size).map(sns.distplot, category).add_legend()
    plt.savefig(outputPath+r'\facetGridPlot_'+ticketId+r'.png')

if __name__ == '__main__':
    data = pd.read_excel(r"D:\Personal\SmartIT\data\diabetes.xlsx")
    distributionPlotter(dataframe=data, ticketId='4567', size=5, hue='Outcome', category='AgeCategory')

