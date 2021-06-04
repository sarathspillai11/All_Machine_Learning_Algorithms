from matplotlib import patches
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import warnings; warnings.simplefilter('ignore')
sns.set_style("white")

def scatterWithEncirclePlotter(df,outputPath,x,y,category,encircleCategoryCol,encircleCategoryValue,description='Bubble Plot with Encircling',ticketId=''):
    print('inside plotter')
    # As many colors as there are unique df['category']
    categories = np.unique(df[category])
    colors = [plt.cm.tab10(i / float(len(categories) - 1)) for i in range(len(categories))]

    # Step 2: Draw Scatterplot with unique color for each category
    fig = plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')

    for i, categor in enumerate(categories):
        plt.scatter(x, y, data=df.loc[df[category] == categor, :], s='dot_size', c=colors[i], label=str(categor), edgecolors='black', linewidths=.5)

    # Step 3: Encircling
    # https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot
    def encircle(x, y, ax=None, **kw):
        if not ax: ax = plt.gca()
        p = np.c_[x, y]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices, :], **kw)
        ax.add_patch(poly)

    # Select data to be encircled
    df_encircle_data = df.loc[df[encircleCategoryCol] == encircleCategoryValue, :]

    # Draw polygon surrounding vertices
    encircle(df_encircle_data.area, df_encircle_data.poptotal, ec="k", fc="gold", alpha=0.1)
    encircle(df_encircle_data.area, df_encircle_data.poptotal, ec="firebrick", fc="none", linewidth=1.5)

    # Step 4: Decorations
    plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000), xlabel=x, ylabel=y)

    plt.xticks(fontsize=12);
    plt.yticks(fontsize=12)
    plt.title(description, fontsize=22)
    plt.legend(fontsize=12)
    plt.savefig(outputPath+r'\scatterEncirclePlot_' + ticketId + r'.png')



if __name__ == '__main__':

    # Step 1: Prepare Data
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")

    scatterWithEncirclePlotter(df, x='area', y='poptotal', category='category', encircleCategoryCol='state', encircleCategoryValue='IN', description='scatter plot with encircled data', ticketId='')