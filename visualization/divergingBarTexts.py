import pandas as pd
import matplotlib.pyplot as plt

def divergingBarTextPlotter(df,outputPath,x,y,description='',ticketId=''):

    x1 = df.loc[:, [x]]
    df['x_z'] = (x1 - x1.mean()) / x1.std()
    df['colors'] = ['red' if x1 < 0 else 'green' for x1 in df['x_z']]
    df.sort_values('x_z', inplace=True)
    df.reset_index(inplace=True)

    # Draw plot
    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=df.index, xmin=0, xmax=df.x_z)
    for x1, y1, tex in zip(df.x_z, df.index, df.x_z):
        t = plt.text(x1, y1, round(tex, 2), horizontalalignment='right' if x1 < 0 else 'left', verticalalignment='center', fontdict={'color': 'red' if x1 < 0 else 'green', 'size': 14})

    # Decorations
    plt.yticks(df.index, df[y], fontsize=12)
    plt.title(description, fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.savefig(outputPath+r'\divergingBarsTextPlot_' + ticketId + r'.png')



if __name__ == '__main__':

    # Prepare Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")
    divergingBarTextPlotter(df,x='mpg',y='cars',description='Diverging Bars ',ticketId='12345')