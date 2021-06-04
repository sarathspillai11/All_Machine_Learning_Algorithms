import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def areaChartPlotter(df,outputPath,timeCol,timeVariantCol,description=' Area chart',ticketId=''):

    x = np.arange(df.shape[0])
    y_returns = ((df[timeVariantCol]).diff().fillna(0) / (df[timeVariantCol]).shift(1)).fillna(0) * 100

    # Plot
    plt.figure(figsize=(16, 10), dpi=80)
    plt.fill_between(x[1:], y_returns[1:], 0, where=y_returns[1:] >= 0, facecolor='green', interpolate=True, alpha=0.7)
    plt.fill_between(x[1:], y_returns[1:], 0, where=y_returns[1:] <= 0, facecolor='red', interpolate=True, alpha=0.7)

    # # Annotate
    # plt.annotate('Peak \n1975', xy=(94.0, 21.0), xytext=(88.0, 28), bbox=dict(boxstyle='square', fc='firebrick'), arrowprops=dict(facecolor='steelblue', shrink=0.05), fontsize=15, color='white')

    # Decorations
    xtickvals = [str(m)[:3].upper() + "-" + str(y) for y, m in zip((df[timeCol]).dt.year, (df[timeCol]).dt.month_name())]
    plt.gca().set_xticks(x[::6])
    #plt.gca().set_xticklabels(xtickvals[::6], rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})
    plt.ylim(-35, 35)
    plt.xlim(1, 100)
    plt.title(description, fontsize=22)
    plt.ylabel(timeVariantCol)
    plt.grid(alpha=0.5)
    plt.savefig(outputPath+r'\areaPlot_' + ticketId + r'.png')


if __name__ == '__main__':

    # Prepare Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/economics.csv", parse_dates=['date']).head(100)
    areaChartPlotter(df,timeCol='date',timeVariantCol='psavert',description=' Area chart',ticketId='12345')
