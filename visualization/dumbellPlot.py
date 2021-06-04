import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.lines as mlines

def dumbellChartPlotter(df,outputPath,timeVariantCol1,timeVariantCol2,description=' dumbell chart',ticketId=''):
    df.sort_values(timeVariantCol2, inplace=True)
    df.reset_index(inplace=True)

    # Func to draw line segment
    def newline(p1, p2, color='black'):
        ax = plt.gca()
        l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color='skyblue')
        ax.add_line(l)
        return l

    # Figure and Axes
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), facecolor='#f7f7f7', dpi=80)

    # Vertical Lines
    ax.vlines(x=.05, ymin=0, ymax=26, color='black', alpha=1, linewidth=1, linestyles='dotted')
    ax.vlines(x=.10, ymin=0, ymax=26, color='black', alpha=1, linewidth=1, linestyles='dotted')
    ax.vlines(x=.15, ymin=0, ymax=26, color='black', alpha=1, linewidth=1, linestyles='dotted')
    ax.vlines(x=.20, ymin=0, ymax=26, color='black', alpha=1, linewidth=1, linestyles='dotted')

    # Points
    ax.scatter(y=df['index'], x=df[timeVariantCol1], s=50, color='#0e668b', alpha=0.7)
    ax.scatter(y=df['index'], x=df[timeVariantCol2], s=50, color='#a3c4dc', alpha=0.7)

    # Line Segments
    for i, p1, p2 in zip(df['index'], df[timeVariantCol1], df[timeVariantCol2]):
        newline([p1, i], [p2, i])

    # Decoration
    ax.set_facecolor('#f7f7f7')
    ax.set_title(description, fontdict={'size': 22})
    ax.set(xlim=(0, .25), ylim=(-1, 27),ylabel = ' Mean change in '+timeVariantCol1+' and '+timeVariantCol2)
    ax.set_xticks([.05, .1, .15, .20])
    ax.set_xticklabels(['5%', '15%', '20%', '25%'])
    ax.set_xticklabels(['5%', '15%', '20%', '25%'])
    plt.show()
    plt.savefig(outputPath+r'\dumbellPlot_' + ticketId + r'.png')


if __name__ == '__main__':

    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/health.csv")
    dumbellChartPlotter(df,timeVariantCol1='pct_2013',timeVariantCol2='pct_2014',description=' dumbell chart', ticketId='123456')
