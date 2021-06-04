from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import pandas as pd
import matplotlib.pyplot as plt

def timeSeriesDecompositionPlotter(df,outputPath,dateColumn,timeVariantColumn,description='time series analysis',ticketId=''):
    dates = pd.DatetimeIndex([parse(d).strftime('%Y-%m-01') for d in df[dateColumn]])
    df.set_index(dates, inplace=True)

    # Decompose
    result = seasonal_decompose(df[timeVariantColumn], model='multiplicative')

    # Plot
    plt.rcParams.update({'figure.figsize': (10, 10)})
    result.plot().suptitle(description)
    #plt.show()
    plt.savefig(outputPath+r'\TimeSeriesDecompositionPlot_' + ticketId + r'.png')


if __name__ == '__main__':


    # Import Data
    df = pd.read_csv('https://github.com/selva86/datasets/raw/master/AirPassengers.csv')

    timeSeriesDecompositionPlotter(df,dateColumn='date',timeVariantColumn='value',description='time series analysis',ticketId='12345')
