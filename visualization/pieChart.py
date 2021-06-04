import seaborn as sns
from matplotlib import pyplot as plt
def pieChart(x,y,ticketId=''):
    # instanciate the figure
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot()

    # ----------------------------------------------------------------------------------------------------
    # plot the data using matplotlib
    ax.pie(y,  # pass the values from our dictionary
           labels=x,  # pass the labels from our dictonary
           autopct='%1.1f%%',  # specify the format to be plotted
           textprops={'fontsize': 10, 'color': "white"}
           # change the font size and the color of the numbers inside the pie
           )
    plt.savefig(r'' + ticketId + r'.png')
if __name__ == '__main__':
    tips = sns.load_dataset("tips")
    x = tips.groupby('day')['total_bill'].agg('sum').reset_index().day
    y = tips.groupby('day')['total_bill'].agg('sum').reset_index().total_bill
    ticketId = '4567'
    pieChart(x , y, ticketId)