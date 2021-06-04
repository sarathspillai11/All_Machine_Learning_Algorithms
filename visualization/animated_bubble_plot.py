import plotly_express as px
from visualization import dataEncoding as dataEncoding
def bubbelplot(dataframe,outputPath, x , y, animation_frame, animation_group, size, color,
               hover_name, log_x, size_max, range_x, range_y, ticketId= ''):

    dataframe = dataEncoding.dataEncoder(dataframe)

    px.scatter(dataframe,outputPath, x, y, animation_frame, animation_group,
               size, color, hover_name, log_x, size_max, range_x, range_y)
    px.savefig(r'' + 'ticketId' + r'.png')
if __name__ == '__main__':
    dataframe = px.data.gapminder()
    x = "gdpPercap"
    y = "lifeExp"
    animation_frame="year"
    animation_group = "country"
    size = "pop"
    color = "country"
    hover_name = "country"
    log_x = True
    size_max = 45
    ticketId = '4567'
    range_x = [100, 100000]
    range_y = [25, 90]
    bubbelplot(dataframe, x , y, animation_frame, animation_group, size, color,
               hover_name, log_x, size_max, range_x, range_y, ticketId)