"""Build a Datapane using historical data."""
import sys
from unittest.mock import Mock

import datapane as dp
import pandas as pd
import plotly.express as px

from coin_test.analysis.data_processing import PricePlotMultiple, PricePlotSingle

if __name__ == "__main__":

    df = px.data.gapminder()

    plotly_chart = px.scatter(
        df.query("year==2007"),
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=60,
    )
    app = dp.App(dp.Plot(plotly_chart)).save(path="plotly-plot.html")

    plot = PricePlotSingle.create(Mock())
    multi_plot = PricePlotMultiple.create(Mock())
    app_nathan = dp.App(
        dp.Page(
            title="Coin-Test Datapane",
            blocks=[
                "### Price Data",
                dp.Group(plot, multi_plot, columns=2),
            ],
        )
    )

    app_nathan.save(path="nathan-plot.html")

    # app = dp.App(
    #     dp.Page(
    #         title="Titanic Dataset",
    #         blocks=[
    #             "### Dataset",
    #             dp.Group(dp.Plot(plot), dp.DataTable(df), columns=2),
    #         ],
    #     ),
    #     dp.Page(
    #         title="Titanic Plot",
    #         blocks=["### Plot", dp.Group(dp.Plot(plot), dp.DataTable(df), columns=2)],
    #     ),
    # )

    # dp.Group(dp.Plot(plot), dp.DataTable(df), columns=2), dp.Group(dp.Plot(plot), dp.DataTable(df), columns=3))
