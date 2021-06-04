import joypy
import seaborn as sns
from matplotlib import pyplot as plt
from visualization import dataEncoding as dataEncoding
def joyPlotter(dataframe,outputPath,x,y,ticketId=''):
    dataframe = dataEncoding.dataEncoder(dataframe)
    joypy.joyplot(dataframe, by=x, column=y,figsize=(5,8))
    plt.savefig(r'' + ticketId + r'.png')
if __name__ == '__main__':
    tips = sns.load_dataset("tips")
    x = "day"
    y = "total_bill"
    ticketId = '4567'
    joyPlotter(tips, x , y, ticketId)