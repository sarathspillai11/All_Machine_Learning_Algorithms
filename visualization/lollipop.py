import pandas as pd
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
import visualization.dataEncoding as dataEncoding
import pandas as pd

def lollipop(dataframe,outputPath,x,y,upperlimit,ticketId=''):
    fig, ax = plt.subplots()

    # Draw the stem and circle
    ax.stem(dataframe[x], dataframe[y], basefmt=' ')

    # Start the graph at 0
    ax.set_ylim(0, upperlimit)
    plt.savefig(ticketId + r'.png')

if __name__ == '__main__':
    data = pd.read_csv("honeyprod.csv")
    data = dataEncoding.dataEncoder(data)
    lollipop(data,x = 'totalprod',y='priceperlb',upperlimit=6, ticketId='4567')

