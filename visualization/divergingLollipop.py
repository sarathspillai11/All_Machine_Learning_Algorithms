import pandas as pd
import matplotlib.pyplot as plt


def divergingLollipopPlotter(df,outputPath,x,y,description='diverging lollipop',ticketId=''):
    x1 = df.loc[:, [x]]
    df['x_z'] = (x1 - x1.mean()) / x1.std()
    df['colors'] = 'black'

    # color fiat differently
    # df.loc[df.cars == 'Fiat X1-9', 'colors'] = 'darkorange'
    df.sort_values('x_z', inplace=True)
    df.reset_index(inplace=True)

    # Draw plot
    import matplotlib.patches as patches

    plt.figure(figsize=(14, 16), dpi=80)
    plt.hlines(y=df.index, xmin=0, xmax=df.x_z, color=df.colors, alpha=0.4, linewidth=1)
    plt.scatter(df.x_z, df.index, color=df.colors, s=300, alpha=0.6)
    plt.yticks(df.index, df[y])
    plt.xticks(fontsize=12)

    # # Annotate
    # plt.annotate('Mercedes Models', xy=(0.0, 11.0), xytext=(1.0, 11), xycoords='data',
    #             fontsize=15, ha='center', va='center',
    #             bbox=dict(boxstyle='square', fc='firebrick'),
    #             arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=1.5', lw=2.0, color='steelblue'), color='white')
    #
    # # Add Patches
    # p1 = patches.Rectangle((-2.0, -1), width=.3, height=3, alpha=.2, facecolor='red')
    # p2 = patches.Rectangle((1.5, 27), width=.8, height=5, alpha=.2, facecolor='green')
    # plt.gca().add_patch(p1)
    # plt.gca().add_patch(p2)

    # Decorate
    plt.title(description, fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(outputPath+r'\divergingLollipopPlot_' + ticketId + r'.png')

if __name__ == '__main__':
    # Prepare Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")
    divergingLollipopPlotter(df,outputPath,x='mpg',y='cars',description='Diverging Lollipop ',ticketId='12345')

