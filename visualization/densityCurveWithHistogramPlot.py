import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def denCurveWithHistoPlotter(df,outputPath,categoryCol,categoryList,categoryVariantCol,description='Density Plot with Histogram',ticketId=''):
    # Draw Plot
    print('inside func')
    plt.figure(figsize=(13, 10), dpi=80)
    colours = ['b','g','r','c','m','y','k','w']
    categories=list(zip(categoryList,colours))
    #print(categories)
    for cat in categories:

        sns.distplot(df.loc[df[categoryCol] == cat[0], categoryVariantCol], color=cat[1], label=cat[0], hist_kws={'alpha': .7}, kde_kws={'linewidth': 3})
    # sns.distplot(df.loc[df['class'] == 'suv', "cty"], color="orange", label="SUV", hist_kws={'alpha': .7}, kde_kws={'linewidth': 3})
    # sns.distplot(df.loc[df['class'] == 'minivan', "cty"], color="g", label="minivan", hist_kws={'alpha': .7}, kde_kws={'linewidth': 3})
    plt.ylim(0, 0.35)

    # Decoration
    plt.title(description, fontsize=22)
    plt.legend()
    plt.savefig(outputPath+r'\densityHistogramPlot_' + ticketId + r'.png')

if __name__ == '__main__':

    # Import Data
    print('started')
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    print('data imported')
    denCurveWithHistoPlotter(df,'class',['compact','suv','minivan'],'cty','123456')

