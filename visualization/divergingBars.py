import pandas as pd
import matplotlib.pyplot as plt


def divergingBarsPlotter(df,outputPath,x,y,description='Diverging Bars ',ticketId=''):
    x1 = df.loc[:, [x]]
    df['x_z'] = (x1 - x1.mean()) / x1.std()
    df['colors'] = ['red' if x1 < 0 else 'green' for x1 in df['x_z']]
    df.sort_values('x_z', inplace=True)
    df.reset_index(inplace=True)


    # Draw plot
    plt.figure(figsize=(14, 10), dpi=80)
    plt.hlines(y=df.index, xmin=0, xmax=df.x_z, color=df.colors, alpha=0.4, linewidth=5)

    # Decorations
    plt.gca().set(ylabel='$Model$', xlabel='$Mileage$')
    plt.yticks(df.index, df[y], fontsize=12)
    plt.title(description, fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(outputPath+r'\divergingBarsPlot_' + ticketId + r'.png')

if __name__ == '__main__':

    # Prepare Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")

    divergingBarsPlotter(df,x='mpg',y='cars',description='Diverging Bars ',ticketId='12345')
