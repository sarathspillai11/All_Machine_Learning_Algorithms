import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import visualization.dataEncoding as dataEncoding
from sklearn.model_selection import train_test_split

def peakTroughTimeSeriesPlotter(df,outputPath,dateColumn,timeVariantColumn,description='time series analysis',ticketId=''):
    # Get the Peaks and Troughs
    data = df[timeVariantColumn].values
    doublediff = np.diff(np.sign(np.diff(data)))
    peak_locations = np.where(doublediff == -2)[0] + 1

    doublediff2 = np.diff(np.sign(np.diff(-1 * data)))
    trough_locations = np.where(doublediff2 == -2)[0] + 1

    # Draw Plot
    plt.figure(figsize=(16, 10), dpi=80)
    plt.plot(dateColumn, timeVariantColumn, data=df, color='tab:blue', label=description)
    plt.scatter((df[dateColumn])[peak_locations], (df[timeVariantColumn])[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
    plt.scatter((df[dateColumn])[trough_locations], (df[timeVariantColumn])[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')

    # Annotate
    for t, p in zip(trough_locations[1::5], peak_locations[::3]):
        plt.text((df[dateColumn])[p], (df[timeVariantColumn])[p] + 15, (df[dateColumn])[p], horizontalalignment='center', color='darkgreen')
        plt.text((df[dateColumn])[t], (df[timeVariantColumn])[t] - 35, (df[dateColumn])[t], horizontalalignment='center', color='darkred')

    # Decoration
    plt.ylim(50, 750)
    xtick_location = df.index.tolist()[::6]
    xtick_labels = (df[dateColumn]).tolist()[::6]

    plt.xticks(xtick_location, xtick_labels, rotation=90, fontsize=12, alpha=.7)
    # plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, alpha=.7)
    plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
    plt.yticks(fontsize=12, alpha=.7)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=.3)
    plt.savefig(outputPath+r'\peakTroughTimeSeriesPlot_' + ticketId + r'.png')

if __name__ == '__main__':

    df = pd.read_csv(r'D:\Personal\SmartIT\data\airPassengers.csv')
    peakTroughTimeSeriesPlotter(df, 'date', 'value', description='time series analysis', ticketId='123456')

# # Get the Peaks and Troughs
# data = df['value'].values
# doublediff = np.diff(np.sign(np.diff(data)))
# peak_locations = np.where(doublediff == -2)[0] + 1
#
# doublediff2 = np.diff(np.sign(np.diff(-1*data)))
# trough_locations = np.where(doublediff2 == -2)[0] + 1
#
# # Draw Plot
# plt.figure(figsize=(16,10), dpi= 80)
# plt.plot('date', 'value', data=df,outputPath, color='tab:blue', label='Air Traffic')
# plt.scatter(df.date[peak_locations], df.value[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
# plt.scatter(df.date[trough_locations], df.value[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')
#
# # Annotate
# for t, p in zip(trough_locations[1::5], peak_locations[::3]):
#     plt.text(df.date[p], df.value[p]+15, df.date[p], horizontalalignment='center', color='darkgreen')
#     plt.text(df.date[t], df.value[t]-35, df.date[t], horizontalalignment='center', color='darkred')
#
# # Decoration
# plt.ylim(50,750)
# xtick_location = df.index.tolist()[::6]
# xtick_labels = df.date.tolist()[::6]
#
# plt.xticks(xtick_location, xtick_labels, rotation=90, fontsize=12, alpha=.7)
# #plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
# plt.yticks(fontsize=12, alpha=.7)
#
# # Lighten borders
# plt.gca().spines["top"].set_alpha(.0)
# plt.gca().spines["bottom"].set_alpha(.3)
# plt.gca().spines["right"].set_alpha(.0)
# plt.gca().spines["left"].set_alpha(.3)
#
# plt.legend(loc='upper left')
# plt.grid(axis='y', alpha=.3)
# plt.savefig(outputPath+r'\peakTroughTimeSeriesPlot_' + '12345' + r'.png')