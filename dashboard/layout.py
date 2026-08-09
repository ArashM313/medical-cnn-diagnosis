"""
Defines the Dash UI layout. No callback logic here -- this file only
describes WHAT is on screen and each component's `id` (callbacks.py wires
up the behavior separately). Keeping these separate means you can restyle
the UI without touching prediction logic, and vice versa.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

import config

MODALITY_OPTIONS = [
    {"label": "Chest X-Ray", "value": "xray"},
    {"label": "CT Scan (Lung)", "value": "ct"},
    {"label": "Brain MRI", "value": "mri"},
]


def build_layout() -> html.Div:
    return dbc.Container(fluid=True, children=[

        # --- Header ---
        dbc.Row(dbc.Col(html.Div([
            html.H2("Medical Image Disease Classifier", className="mt-3"),
            html.P("Upload an X-ray, CT, or MRI scan for CNN-based classification "
                   "with Grad-CAM visual explanation.", className="text-muted"),
        ]))),

        html.Hr(),

        dbc.Tabs(id="main-tabs", active_tab="tab-diagnose", children=[

            # ================= TAB 1: Diagnose =================
            dbc.Tab(label="Diagnose", tab_id="tab-diagnose", children=[
                dbc.Row(className="mt-4", children=[

                    # --- Left column: controls + upload ---
                    dbc.Col(width=4, children=[
                        html.Label("1. Select scan type"),
                        dcc.Dropdown(
                            id="modality-dropdown",
                            options=MODALITY_OPTIONS,
                            value="xray",
                            clearable=False,
                        ),
                        html.Br(),

                        html.Label("2. Upload image"),
                        dcc.Upload(
                            id="upload-image",
                            children=html.Div([
                                "Drag and drop or ", html.A("select a scan image")
                            ]),
                            style={
                                "width": "100%", "height": "80px", "lineHeight": "80px",
                                "borderWidth": "1px", "borderStyle": "dashed",
                                "borderRadius": "8px", "textAlign": "center",
                            },
                            accept="image/*",
                            multiple=False,
                        ),
                        html.Br(),

                        dbc.Button("Run Diagnosis", id="predict-button",
                                   color="primary", className="w-100", disabled=True),

                        html.Div(id="upload-status", className="mt-2 text-muted small"),
                    ]),

                    # --- Right column: images + results ---
                    dbc.Col(width=8, children=[
                        dbc.Row([
                            dbc.Col(width=6, children=[
                                html.H6("Uploaded Scan"),
                                html.Img(id="original-image-display",
                                         style={"width": "100%", "borderRadius": "8px"}),
                            ]),
                            dbc.Col(width=6, children=[
                                html.H6("Grad-CAM Explanation"),
                                dcc.Loading(html.Img(
                                    id="gradcam-image-display",
                                    style={"width": "100%", "borderRadius": "8px"},
                                )),
                            ]),
                        ]),

                        html.Hr(),

                        dcc.Loading(html.Div(id="prediction-result")),
                        dcc.Graph(id="probability-bar-chart"),
                    ]),
                ]),
            ]),

            # ================= TAB 2: Model Performance =================
            dbc.Tab(label="Model Performance", tab_id="tab-metrics", children=[
                dbc.Row(className="mt-4", children=[
                    dbc.Col(width=4, children=[
                        html.Label("Select model to inspect"),
                        dcc.Dropdown(
                            id="metrics-modality-dropdown",
                            options=MODALITY_OPTIONS,
                            value="xray",
                            clearable=False,
                        ),
                    ]),
                ]),

                html.Div(id="metrics-summary", className="mt-3"),

                dbc.Row([
                    dbc.Col(width=6, children=[
                        html.H6("Confusion Matrix (Test Set)"),
                        dcc.Graph(id="confusion-matrix-chart"),
                    ]),
                    dbc.Col(width=6, children=[
                        html.H6("Per-Class Precision / Recall / F1"),
                        html.Div(id="per-class-metrics-table"),
                    ]),
                ]),
            ]),
        ]),

        # Hidden store: holds the raw uploaded image bytes (base64) so the
        # predict-button callback can access it without re-reading the
        # upload component directly (keeps callback inputs cleaner).
        dcc.Store(id="uploaded-image-store"),
    ])