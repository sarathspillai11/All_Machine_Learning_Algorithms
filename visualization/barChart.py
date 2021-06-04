import seaborn as sns
from matplotlib import pyplot as plt
def barChart(x,y,ticketId=''):
    # instanciate the figure
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(x, y)
    plt.show()
    plt.savefig(r'' + ticketId + r'.png')
if __name__ == '__main__':
    tips = sns.load_dataset("tips")
    x = tips.groupby('day')['total_bill'].agg('sum').reset_index().day
    y = tips.groupby('day')['total_bill'].agg('sum').reset_index().total_bill
    ticketId = '4567'
    barChart(x , y, ticketId)