"""
Entry point for the dashboard. Run this file directly in PyCharm to start
the local web server, then open the printed URL (typically
http://127.0.0.1:8050) in your browser.
"""

import dash
import dash_bootstrap_components as dbc

from dashboard.callbacks import register_callbacks
from dashboard.layout import build_layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Medical Image Classifier",
)

app.layout = build_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True)