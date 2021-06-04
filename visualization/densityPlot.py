import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def densityPlotter(df,outputPath,categoryCol,categoryList,categoryVariantCol,description='Density Plot',ticketId=''):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    categories = list(zip(categoryList, colours))
    # Draw Plot
    plt.figure(figsize=(16, 10), dpi=80)
    for cat in categories:

        sns.kdeplot(df.loc[df[categoryCol] == cat[0], categoryVariantCol], shade=True, color=cat[1], label=categoryCol+"="+str(cat[0]), alpha=.7)


    # Decoration
    plt.title(description, fontsize=22)
    plt.legend()
    plt.savefig(outputPath+r'\densityPlot_' + ticketId + r'.png')

if __name__ == '__main__':

    # Import Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    densityPlotter(df, 'cyl', [4,5,6,8], 'cty', '123456')

