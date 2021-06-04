import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def slopePlotter(df,outputPath,yearColumn1,yearColumn2,timeVariantColumn,description='slope chart',ticketId=''):

    left_label = [str(c) + ', ' + str(round(y)) for c, y in zip(df[timeVariantColumn], df[yearColumn1])]
    right_label = [str(c) + ', ' + str(round(y)) for c, y in zip(df[timeVariantColumn], df[yearColumn2])]
    klass = ['red' if (y1 - y2) < 0 else 'green' for y1, y2 in zip(df[yearColumn1], df[yearColumn2])]

    # draw line
    # https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib/36479941
    def newline(p1, p2, color='black'):
        ax = plt.gca()
        l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color='red' if p1[1] - p2[1] > 0 else 'green', marker='o', markersize=6)
        ax.add_line(l)
        return l

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), dpi=80)

    # Vertical Lines
    ax.vlines(x=1, ymin=500, ymax=13000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
    ax.vlines(x=3, ymin=500, ymax=13000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

    # Points
    ax.scatter(y=df[yearColumn1], x=np.repeat(1, df.shape[0]), s=10, color='black', alpha=0.7)
    ax.scatter(y=df[yearColumn2], x=np.repeat(3, df.shape[0]), s=10, color='black', alpha=0.7)

    # Line Segmentsand Annotation
    for p1, p2, c in zip(df[yearColumn1], df[yearColumn2], df[timeVariantColumn]):
        newline([1, p1], [3, p2])
        ax.text(1 - 0.05, p1, c + ', ' + str(round(p1)), horizontalalignment='right', verticalalignment='center', fontdict={'size': 14})
        ax.text(3 + 0.05, p2, c + ', ' + str(round(p2)), horizontalalignment='left', verticalalignment='center', fontdict={'size': 14})

    yearvals = list(df[yearColumn1])
    yearvals.extend(list(df[yearColumn2]))
    yMax = max(yearvals)
    yMax = int(math.ceil(yMax / 100.0)) * 100
    # 'Before' and 'After' Annotations
    ax.text(1 - 0.05, yMax, 'BEFORE', horizontalalignment='right', verticalalignment='center', fontdict={'size': 18, 'weight': 700})
    ax.text(3 + 0.05, yMax, 'AFTER', horizontalalignment='left', verticalalignment='center', fontdict={'size': 18, 'weight': 700})

    # Decoration
    ax.set_title(description, fontdict={'size': 22})
    ax.set(xlim=(0, 4), ylim=(0, yMax))
    ax.set_xticks([1, 3])
    ax.set_xticklabels([yearColumn1, yearColumn2])
    plt.yticks(np.arange(500, yMax, 2000), fontsize=12)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.0)
    #plt.show()
    plt.savefig(outputPath+r'\slopPlot_' + ticketId + r'.png')

if __name__ == '__main__':
    # Import Data
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/gdppercap.csv")
    slopePlotter(df,yearColumn1='1952',yearColumn2='1957',timeVariantColumn='continent',description='slope chart',ticketId='12345')


