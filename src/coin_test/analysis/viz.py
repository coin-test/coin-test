"""Build a Datapane using historical data."""
import sys
from unittest.mock import Mock

import datapane as dp
import pandas as pd
import plotly.express as px

from coin_test.analysis.data_processing import (
    PlotParameters,
    PricePlotMultiple,
    PricePlotSingle,
)

if __name__ == "__main__":

    plot_params = PlotParameters()

    plot = PricePlotSingle.create(Mock(), plot_params)
    multi_plot = PricePlotMultiple.create(Mock(), plot_params)
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
