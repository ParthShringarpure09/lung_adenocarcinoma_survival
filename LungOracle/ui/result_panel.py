"""
Reusable prediction result panel.
"""

import gradio as gr
from ui.prediction import get_prediction_placeholder


def build_result_panel(prefix=""):

    risk_card = gr.HTML(
        value=get_prediction_placeholder()
    )

    km_plot = gr.Plot(
        visible=False
    )

    shap_plot = gr.Plot(
        visible=False
    )

    explanation = gr.Textbox(
        label="Model Interpretation",
        interactive=False,
        lines=2,
        visible=False,
    )

    immune_plot = gr.Plot(
        visible=False
    )

    return (
        risk_card,
        km_plot,
        shap_plot,
        immune_plot,
        explanation,
    )