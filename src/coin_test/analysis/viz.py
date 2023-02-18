import altair as alt
from vega_datasets import data
import datapane as dp
import pandas as pd
import sys

if(__name__ == "__main__"):
#   modulename = 'datapane'
#   if modulename not in sys.modules:
#       print(f'You have not imported the {modulename} module')

#   df = data.iris()
#   fig = (
#       alt.Chart(df)
#       .mark_point()
#       .encode(x="petalLength:Q", y="petalWidth:Q", color="species:N")
#   )
#   app = dp.App(dp.Plot(fig), dp.DataTable(df))
#   app.save(path="my_app.html")

    alt.data_transformers.disable_max_rows()

    dataset = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
    df = (
        dataset.groupby(["continent", "date"])["new_cases_smoothed_per_million"]
        .mean()
        .reset_index()
    )

    plot = (
        alt.Chart(df)
        .mark_area(opacity=0.4, stroke="black")
        .encode(
            x="date:T",
            y=alt.Y("new_cases_smoothed_per_million:Q", stack=None),
            color=alt.Color("continent:N", scale=alt.Scale(scheme="set1")),
            tooltip="continent:N",
        )
        .interactive()
        .properties(width="container")
    )

    app = dp.App(    dp.Page(title="Titanic Dataset", blocks=["### Dataset", dp.Group(dp.Plot(plot), dp.DataTable(df), columns=2)]),
        dp.Page(title="Titanic Plot", blocks=["### Plot", dp.Group(dp.Plot(plot), dp.DataTable(df), columns=2)])
    )
        
        #dp.Group(dp.Plot(plot), dp.DataTable(df), columns=2), dp.Group(dp.Plot(plot), dp.DataTable(df), columns=3))

    
    
    app = app.save(path="grid-layout.html")