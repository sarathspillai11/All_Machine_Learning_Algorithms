from dateutil.parser import parse
import matplotlib.pyplot as plt
import pandas as pd


def seasonalPlotter(df,outputPath,dateColumn,timeVariantColumn,description='Monthly seasonal plot',ticketId=''):
    # Prepare data
    print(df.columns)
    df['year'] = [parse(d).year for d in df[dateColumn]]
    df['month'] = [parse(d).strftime('%b') for d in df[dateColumn]]
    years = df['year'].unique()

    # Draw Plot
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive', 'deeppink', 'steelblue', 'firebrick', 'mediumseagreen']
    plt.figure(figsize=(16, 10), dpi=80)

    for i, y in enumerate(years):
        plt.plot('month', timeVariantColumn, data=df.loc[df.year == y, :], color=mycolors[i], label=y)
        plt.text(df.loc[df.year == y, :].shape[0] - .9, df.loc[df.year == y, timeVariantColumn][-1:].values[0], y, fontsize=12, color=mycolors[i])

    # Decoration
    plt.ylim(50, 750)
    plt.xlim(-0.3, 11)
    # plt.ylabel('$Air Traffic$')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title(description, fontsize=22)
    plt.grid(axis='y', alpha=.3)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.5)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.5)
    # plt.legend(loc='upper right', ncol=2, fontsize=12)
    # plt.show()
    plt.savefig(outputPath+r'\seasonalPlot_' + ticketId + r'.png')

if __name__ == '__main__':

    # Import Data
    df = pd.read_csv('https://github.com/selva86/datasets/raw/master/AirPassengers.csv')
    seasonalPlotter(df,'date','value',description='Monthly seasonal plot',ticketId='12345')

