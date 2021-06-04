import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def dotBoxPlotter(df,outputPath,x,y,hue,description='Dot + Box Plot',ticketId=''):
    # Draw Plot
    plt.figure(figsize=(13, 10), dpi=80)
    sns.boxplot(x=x, y=y, data=df, hue=hue)
    sns.stripplot(x=x, y=y, data=df, color='black', size=3, jitter=1)

    for i in range(len(df[x].unique()) - 1):
        plt.vlines(i + .5, 10, 45, linestyles='solid', colors='gray', alpha=0.2)

    # Decoration
    plt.title(description, fontsize=22)
    plt.legend(title=hue)
    plt.savefig(outputPath+r'\dotBoxPlot_' + ticketId + r'.png')



if __name__ == '__main__':

    # Import Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    dotBoxPlotter(df,x='class',y='hwy',hue='cyl',ticketId='123456')

