import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def  distributedDotPlot(df_raw,outputPath,x,y,category,categoryList,description='Distributed dot plot',ticketId=''):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    categories = dict(zip(categoryList, colours))
    #cyl_colors = {4: 'tab:red', 5: 'tab:green', 6: 'tab:blue', 8: 'tab:orange'}
    df_raw['_color'] = df_raw[category].map(categories)

    # Mean and Median city mileage by make
    df = df_raw[[x, y]].groupby(y).apply(lambda x: x.mean())
    df.sort_values(x, ascending=False, inplace=True)
    df.reset_index(inplace=True)
    df_median = df_raw[[x, y]].groupby(y).apply(lambda x: x.median())

    # Draw horizontal lines
    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    ax.hlines(y=df.index, xmin=0, xmax=40, color='gray', alpha=0.5, linewidth=.5, linestyles='dashdot')

    # Draw the Dots
    for i, make in enumerate(df[y]):
        df_make = df_raw.loc[df_raw[y] == make, :]
        ax.scatter(y=np.repeat(i, df_make.shape[0]), x=x, data=df_make, s=75, edgecolors='gray', c='w', alpha=0.5)
        ax.scatter(y=i, x=x, data=df_median.loc[df_median.index == make, :], s=75, c='firebrick')

    # Annotate
    ax.text(33, 13, "$red \; dots \; are \; the \: median$", fontdict={'size': 12}, color='firebrick')

    # Decorations
    red_patch = plt.plot([], [], marker="o", ms=10, ls="", mec=None, color='firebrick', label="Median")
    plt.legend(handles=red_patch)
    ax.set_title(description, fontdict={'size': 22})
    ax.set_xlabel(x, alpha=0.7)
    ax.set_yticks(df.index)
    ax.set_yticklabels(df[y].str.title(), fontdict={'horizontalalignment': 'right'}, alpha=0.7)
    ax.set_xlim(1, 40)
    plt.xticks(alpha=0.7)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.grid(axis='both', alpha=.4, linewidth=.1)
    plt.savefig(outputPath+r'\distributedDotPlot_' + ticketId + r'.png')

if __name__ == '__main__':


    # Prepare Data
    df_raw = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    distributedDotPlot(df_raw,x='cty',y='manufacturer',category='cyl',categoryList=[4,5,6,8],ticketId='123456')