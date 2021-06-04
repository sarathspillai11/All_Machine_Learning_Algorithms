from pywaffle import Waffle
import seaborn as sns
from matplotlib import pyplot as plt
from visualization import dataEncoding as dataEncoding
def joyPlotter(dataframe,outputPath,x,rows,columns,ticketId=''):
    dataframe = dataEncoding.dataEncoder(dataframe)
    # plot the data using pywaffle
    plt.figure(
        FigureClass=Waffle,
        rows=rows,
        columns=columns,
        values=dataframe['size'],
        legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1), "fontsize": "12"},
        figsize=(20, 7)
    )

    plt.savefig(r'' + ticketId + r'.png')
if __name__ == '__main__':
    tips = sns.load_dataset("tips")
    x = "day"
    ticketId = '4567'
    rows=6
    columns=1
    joyPlotter(tips, x ,rows,columns, ticketId)