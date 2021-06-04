import pandas as pd
import matplotlib.pyplot as plt


def divergingDotPlotter(df,outputPath,x,y,description='diverging dots',ticketId=''):
    x1 = df.loc[:, [x]]
    df['x_z'] = (x1 - x1.mean()) / x1.std()
    df['colors'] = ['red' if x1 < 0 else 'darkgreen' for x1 in df['x_z']]
    df.sort_values('x_z', inplace=True)
    df.reset_index(inplace=True)

    # Draw plot
    plt.figure(figsize=(14, 16), dpi=80)
    plt.scatter(df.x_z, df.index, s=450, alpha=.6, color=df.colors)
    for x1, y1, tex in zip(df.x_z, df.index, df.x_z):
        t = plt.text(x1, y1, round(tex, 1), horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white'})

    # Decorations
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.3)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.3)
    plt.gca().spines["left"].set_alpha(.3)

    plt.yticks(df.index, df[y])
    plt.title(description, fontdict={'size': 20})
    plt.xlabel('$Mileage$')
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-2.5, 2.5)

    plt.savefig(outputPath+r'\divergingDotsPlot_' + ticketId + r'.png')

if __name__ == '__main__':
    # Prepare Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")
    divergingDotPlotter(df, x='mpg', y='cars', description='Diverging Dots ', ticketId='12345')