"""
Predict Mode tab.

Allows users to upload their own RNA-seq profile
and generate a personalised survival prediction.
"""

from dataclasses import dataclass

import gradio as gr

from ui.sidebar import (
    get_sidebar_heading,
    get_sidebar_stats,
    get_disclaimer,
)

from ui.prediction import get_empty_prediction


# ============================================================
# UI References
# ============================================================

@dataclass
class PredictTab:

    upload: gr.File

    predict_btn: gr.Button

    risk: gr.HTML

    km: gr.Plot

    shap: gr.Plot

    immune: gr.Plot

    explanation: gr.Textbox


# ============================================================
# Builder
# ============================================================

def build_predict_tab() -> PredictTab:

    with gr.Tab("Predict"):

        with gr.Row(elem_classes=["main-pad"]):

            # ===================================================
            # Sidebar
            # ===================================================

            with gr.Column(
                scale=1,
                min_width=320,
                elem_classes=[
                    "sidebar-col",
                    "workspace-card"
                ]
            ):

                gr.HTML(get_sidebar_heading())

                upload = gr.File(
                    label="Gene Expression CSV",
                    file_types=[".csv"]
                )

                predict_btn = gr.Button(
                    "Predict Survival",
                    variant="primary",
                    elem_id="predict-analyse-btn"
                )

                gr.HTML(get_sidebar_stats())

                gr.HTML(get_disclaimer())

            # ===================================================
            # Results
            # ===================================================

            with gr.Column(
                scale=3,
                elem_classes=["main-content"]
            ):

                risk = gr.HTML(
                    value=get_empty_prediction()
                )

                with gr.Row():

                    km = gr.Plot(
                        label="Kaplan-Meier Curve",
                        visible=False,
                        elem_classes=["lo-card"]
                    )

                    shap = gr.Plot(
                        label="SHAP Explanation",
                        visible=False,
                        elem_classes=["lo-card"]
                    )

                immune = gr.Plot(
                    label="Immune Landscape",
                    visible=False,
                    elem_classes=["lo-card"]
                )

                explanation = gr.Textbox(
                    label="Model Interpretation",
                    interactive=False,
                    lines=5,
                    visible=False,
                    elem_classes=["lo-card"]
                )

    return PredictTab(

        upload=upload,

        predict_btn=predict_btn,

        risk=risk,

        km=km,

        shap=shap,

        immune=immune,

        explanation=explanation

    )