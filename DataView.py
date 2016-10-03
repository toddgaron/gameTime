from os.path import dirname, join

import numpy as np
import pandas.io.sql as psql
import sqlite3 as sql
import pandas as pd
import pickle

from bokeh.plotting import figure
from bokeh.charts import Histogram
from bokeh.layouts import layout, widgetbox, row, column
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import push_notebook, show, output_notebook, curdoc
from bokeh.palettes import Plasma9
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.embed import autoload_server
from bokeh.client import push_session

#output_notebook()


games=pickle.load(open( 'Parsedgames.p', "rb" ) )
games2=filter(lambda x: x[15]>200,games)
games2.sort(key=lambda x: int(x[0]))
print games2[0]

Games=pd.DataFrame(games2,columns=['id','name','year','minplayers'
                             ,'maxplayers','mintime','maxtime','playtime'
                             ,'avgage','avglang','cat','mech','numrevs'
                             ,'rating','std','own','trad','wish','comments','description'
                    ])
Games['mintime']=Games['mintime']/60.
Games['maxtime']=Games['maxtime']/60.
Games['playtime']=Games['playtime']/60.
Games['maxplayers']=Games['maxplayers'].apply(lambda x: min([x,10]))
gmax=max(Games['rating'])
gmin=min(Games['rating'])
Games['alpha']=map(lambda x: x/gmax, Games['rating'])
Games['color'] = map(lambda x: Plasma9[int(round(8 * (x-gmin)/(gmax-gmin),0))], Games['rating'])#map(lambda x: tuple(map(lambda y: 255*y, m.to_rgba(x)[:3])), Games['rating'])
Games.head()

axis_map = {
    "Numeric Rating": "rating",
    "Number of Reviews": "numrevs",
    "Owners": "own",
    "Wanters": "wish",
    "Year": "year",
}

#desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)

# Create Input controls
reviews = Slider(title="Minimum number of reviews", value=1000, start=100, end=30000, step=100)
min_year = Slider(title="Year released", start=1900, end=2016, value=1940, step=1)
max_year = Slider(title="End Year released", start=1900, end=2016, value=2016, step=1)
min_players = Slider(title="Min Number of Players", start=1, end=10, value=2, step=1)
max_players = Slider(title="Max Number of Players", start=1, end=10, value=5, step=1)
# oscars = Slider(title="Minimum number of Oscar wins", start=0, end=4, value=0, step=1)
# boxoffice = Slider(title="Dollars at Box Office (millions)", start=0, end=800, value=0, step=1)
# genre = Select(title="Genre", value="All",
#                options=open(join(dirname(__file__), 'genres.txt')).read().split())
# director = TextInput(title="Director name contains")
# cast = TextInput(title="Cast names contains")
x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="Numeric Rating")
y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="Number of Reviews")

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], color=[], title=[], year=[], alpha=[]))


hover = HoverTool(tooltips=[
    ("Title", "@title"),
    ("Year", "@year"),
    ("Rating","@rating")
])

p = figure(plot_height=600, plot_width=700, title="", toolbar_location=None, tools=[hover])
p.circle(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")

## create the horizontal histogram
# hhist, hedges = np.histogram(source.data['x'] , bins=20)
# hzeros = np.zeros(len(hedges)-1)
# hmax = max(hhist)*1.1
# 
# LINE_ARGS = dict(color="#3A5785", line_color=None)
# 
# ph = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200, x_range=p.x_range,
#             y_range=(-hmax, hmax), min_border=10, min_border_left=50, y_axis_location="right")
# ph.xgrid.grid_line_color = None
# ph.yaxis.major_label_orientation = np.pi/4
# ph.background_fill_color = "#fafafa"
# 
# ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="#3A5785")
# hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
# hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)


def select_games():
#     genre_val = genre.value
#     director_val = director.value.strip()
#     cast_val = cast.value.strip()
    selected = Games[
        (Games.numrevs >= reviews.value) &
        (Games.minplayers >= min_players.value) &
        (Games.maxplayers <= max_players.value) &
        (Games.year >= min_year.value) &
        (Games.year <= max_year.value) #&
        # (Games.BoxOffice >= (boxoffice.value * 1e6)) & (Games.Oscars >= oscars.value)
    ]
#     if (genre_val != "All"):
#         selected = selected[selected.Genre.str.contains(genre_val)==True]
#     if (director_val != ""):
#         selected = selected[selected.Director.str.contains(director_val)==True]
#     if (cast_val != ""):
#         selected = selected[selected.Cast.str.contains(cast_val)==True]
    return selected


def update():
    df = select_games()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "%d games selected" % len(df)
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        color=df["color"],
        title=df["name"],
        year=df["year"],
        rating=df["rating"],
        alpha=df["alpha"],
    )

controls = [reviews, min_year, max_year, min_players, max_players, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'scale_width'  # 'scale_width' also looks nice with this example

inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    #['desc'],
    [inputs, p],
], sizing_mode=sizing_mode)

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Games"
# session = push_session(curdoc())
# 
# html = """
# <html>
#   <head></head>
#   <body>
#     %s
#   </body>
# </html>
# """ % autoload_server(l, session_id=session.id)
# 
# with open("animated.html", "w+") as f:
#     f.write(html)
# 
# 
# curdoc().add_periodic_callback(update, 30)
# 
# session.loop_until_closed()