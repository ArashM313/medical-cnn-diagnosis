"""
All interactive behavior lives here. Two independent callback groups:

  1. Diagnose tab: upload image -> enable predict button -> on click, run
     the modality's model + Grad-CAM -> display results.
  2. Model Performance tab: modality dropdown -> load that modality's
     precomputed metrics JSON (from training/evaluate.py) -> render charts.

Models are loaded lazily and CACHED in memory (_MODEL_CACHE) so switching
between predictions on the same modality doesn't reload weights from disk
every time -- only the first prediction per modality pays that cost.
"""

import base64
import json
import os
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
import torch
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
from PIL import Image

import config
from interpretability.gradcam import explain_prediction
from models.cnn_model import build_model
from utils.preprocessing import get_transforms

_MODEL_CACHE = {}  # modality -> (model, classes, decision_threshold)


def _get_model(modality: str):
    """Load a modality's model once, cache it, and reuse across predictions."""
    if modality in _MODEL_CACHE:
        return _MODEL_CACHE[modality]

    cfg = config.MODALITIES[modality]
    if not os.path.exists(cfg["checkpoint"]):
        raise FileNotFoundError(
            f"No trained checkpoint for '{modality}' yet. Run training/train.py first."
        )

    checkpoint = torch.load(cfg["checkpoint"], map_location=config.DEVICE)
    classes = checkpoint["classes"]
    decision_threshold = checkpoint.get("decision_threshold", 0.5)
    use_pretrained = bool(cfg.get("use_pretrained", False))

    model = build_model(num_classes=len(classes), device=config.DEVICE, pretrained=use_pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _MODEL_CACHE[modality] = (model, classes, decision_threshold)
    return model, classes, decision_threshold


def _pil_to_data_uri(pil_image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URI, for use as an html.Img src."""
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _decode_upload(contents: str) -> Image.Image:
    """dcc.Upload gives contents as 'data:image/...;base64,<data>' -- decode to a PIL Image."""
    header, encoded = contents.split(",", 1)
    decoded_bytes = base64.b64decode(encoded)
    return Image.open(BytesIO(decoded_bytes)).convert("RGB")


def register_callbacks(app):

    # ---------- Upload handling: preview image + enable predict button ----------
    @app.callback(
        Output("original-image-display", "src"),
        Output("upload-status", "children"),
        Output("predict-button", "disabled"),
        Output("uploaded-image-store", "data"),
        Input("upload-image", "contents"),
        State("upload-image", "filename"),
    )
    def on_upload(contents, filename):
        if contents is None:
            return None, "", True, None

        try:
            pil_image = _decode_upload(contents)
        except Exception:
            return None, "Could not read that file as an image. Please upload a valid image.", True, None

        preview_src = _pil_to_data_uri(pil_image)
        status = f"Loaded: {filename}"
        return preview_src, status, False, contents

    # ---------- Predict button: run model + Grad-CAM ----------
    @app.callback(
        Output("gradcam-image-display", "src"),
        Output("prediction-result", "children"),
        Output("probability-bar-chart", "figure"),
        Input("predict-button", "n_clicks"),
        State("uploaded-image-store", "data"),
        State("modality-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_predict(n_clicks, stored_contents, modality):
        if stored_contents is None:
            return None, dbc_alert("Please upload an image first.", "warning"), go.Figure()

        try:
            model, classes, decision_threshold = _get_model(modality)
        except FileNotFoundError as e:
            return None, dbc_alert(str(e), "danger"), go.Figure()

        pil_image = _decode_upload(stored_contents)
        eval_transform = get_transforms(config.IMAGE_SIZE, train=False)

        result = explain_prediction(model, pil_image, eval_transform, config.DEVICE, decision_threshold=decision_threshold)

        predicted_class = classes[result["predicted_class_index"]]
        confidence_pct = result["confidence"] * 100

        result_card = dbc_alert(
            [
                html.H5(f"Prediction: {predicted_class}", className="mb-1"),
                html.P(f"Confidence: {confidence_pct:.1f}%", className="mb-0"),
            ],
            "success" if confidence_pct >= 70 else "warning",
        )

        fig = go.Figure(go.Bar(
            x=result["all_probabilities"] * 100,
            y=classes,
            orientation="h",
            marker_color="#0d6efd",
        ))
        fig.update_layout(
            title="Class Probabilities (%)",
            xaxis_title="Probability (%)",
            xaxis_range=[0, 100],
            margin=dict(l=120, r=20, t=40, b=40),
            height=250,
        )

        gradcam_src = _pil_to_data_uri(result["overlay_image"])
        return gradcam_src, result_card, fig

    # ---------- Model Performance tab ----------
    @app.callback(
        Output("metrics-summary", "children"),
        Output("confusion-matrix-chart", "figure"),
        Output("per-class-metrics-table", "children"),
        Input("metrics-modality-dropdown", "value"),
    )
    def on_metrics_modality_change(modality):
        metrics_path = os.path.join(config.CHECKPOINT_DIR, f"{modality}_metrics.json")

        if not os.path.exists(metrics_path):
            msg = dbc_alert(
                f"No evaluation metrics yet for '{modality}'. "
                f"Run training/evaluate.py after training this modality.",
                "info",
            )
            return msg, go.Figure(), None

        with open(metrics_path) as f:
            report = json.load(f)

        summary = dbc_alert(
            [
                html.H5(f"{modality.upper()} — Test Accuracy: {report['test_accuracy']*100:.2f}%"),
                html.P(
                    f"Macro F1: {report['macro_f1']:.3f}  |  "
                    f"ROC-AUC: {report['roc_auc_ovr']:.3f}" if report["roc_auc_ovr"] is not None
                    else f"Macro F1: {report['macro_f1']:.3f}  |  ROC-AUC: N/A",
                    className="mb-0",
                ),
                html.Small(f"Evaluated on {report['test_set_size']} held-out test images "
                           f"(checkpoint from epoch {report['checkpoint_epoch']})",
                           className="text-muted"),
            ],
            "light",
        )

        cm = np.array(report["confusion_matrix"])
        classes = report["classes"]
        cm_fig = go.Figure(go.Heatmap(
            z=cm, x=classes, y=classes,
            colorscale="Blues", text=cm, texttemplate="%{text}",
        ))
        cm_fig.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            margin=dict(l=80, r=20, t=20, b=80), height=350,
        )

        rows = []
        for cls, stats in report["per_class"].items():
            rows.append(html.Tr([
                html.Td(cls),
                html.Td(f"{stats['precision']:.3f}"),
                html.Td(f"{stats['recall']:.3f}"),
                html.Td(f"{stats['f1']:.3f}"),
                html.Td(stats["support"]),
            ]))

        table = html.Table(className="table table-sm", children=[
            html.Thead(html.Tr([
                html.Th("Class"), html.Th("Precision"), html.Th("Recall"),
                html.Th("F1"), html.Th("Support"),
            ])),
            html.Tbody(rows),
        ])

        return summary, cm_fig, table


def dbc_alert(children, color):
    """Small wrapper around dbc.Alert kept at module scope for reuse above."""
    return dbc.Alert(children, color=color)