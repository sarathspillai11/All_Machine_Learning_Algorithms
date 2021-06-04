import plotly.express as px
from plotly.offline import init_notebook_mode, plot_mpl
def treeMap(df,outputPath,x,y,ticketId=''):
    # instanciate the figure
    fig = px.treemap(df,outputPath, path=x, values=y)
    plot_mpl(fig)
    # If you want to to download an image of the figure as well
    import plotly.io as pio
    pio.write_image(fig, 'fig1.png')
if __name__ == '__main__':
    df = px.data.tips()
    x = ['day', 'time', 'sex']
    y = 'total_bill'
    ticketId = '4567'
    treeMap(df,x, y, ticketId)