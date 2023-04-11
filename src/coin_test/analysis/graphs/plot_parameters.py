"""Plot parameters object."""

import os

import plotly.graph_objects as go


class PlotParameters:
    """Plot parameters to pass to each plot."""

    def __init__(
        self,
        asset_dir: str,
        line_styles: tuple[str, ...] = ("solid", "dash", "dot", "dashdot"),
        line_colors: tuple[str, ...] = (
            "rgb(0, 0, 0)",
            "rgb(230, 159, 0)",
            "rgb(86, 180, 233)",
            "rgb(0, 158, 115)",
            "rgb(0, 158, 115)",
            "rgb(0, 114, 178)",
        ),
        label_font_color: str = "black",
        label_font_family: str = "Courier New, monospace",
        title_font_size: int = 18,
        axes_font_size: int = 10,
        line_width: int = 2,
    ) -> None:
        """Initialize plot parameters."""
        os.makedirs(asset_dir, exist_ok=True)
        self.asset_dir = asset_dir
        self.line_styles = line_styles
        self.line_colors = line_colors
        self.line_width = line_width
        self.title_font = dict(
            family=label_font_family, size=title_font_size, color=label_font_color
        )
        self.axes_font = dict(
            family=label_font_family, size=axes_font_size, color=label_font_color
        )
        self.legend_font = self.axes_font

    @staticmethod
    def update_plotly_fig(
        plot_params: "PlotParameters",
        fig: go.Figure,
        title: str | None,
        x_lbl: str,
        y_lbl: str,
        legend_title: str = "",
    ) -> None:
        """Update Plotly figure."""
        fig.update_layout(
            title={"text": title, "font": plot_params.title_font},
            xaxis_title={"text": x_lbl, "font": plot_params.axes_font},
            yaxis_title={"text": y_lbl, "font": plot_params.axes_font},
            legend_title={"text": legend_title, "font": plot_params.legend_font},
        )

    def compress_fig(self, fig: go.Figure, name: str, f: str = "png") -> str:
        """Compress figure."""
        fig.update_layout()
        path = os.path.join(self.asset_dir, f"{name}.{f}")
        i = 0
        while os.path.exists(path):
            i += 1
            path = os.path.join(self.asset_dir, f"{name}({i}).{f}")
        fig.write_image(path, scale=2, width=1000, height=450)
        return path
