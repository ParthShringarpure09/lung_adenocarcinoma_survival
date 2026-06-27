"""
Demo Patients tab.

Builds the entire demo interface but does not register callbacks.
"""

from dataclasses import dataclass

import gradio as gr

from ui.sidebar import (
    get_sidebar_heading,
    get_sidebar_stats,
    get_disclaimer,
)

from ui.prediction import get_prediction_placeholder


# ============================================================
# UI References
# ============================================================

@dataclass
class DemoTab:

    dropdown: gr.Dropdown

    story: gr.HTML

    analyse_btn: gr.Button

    risk: gr.HTML

    km: gr.Plot

    shap: gr.Plot

    immune: gr.Plot

    explanation: gr.Textbox


# ============================================================
# Builder
# ============================================================

def build_demo_tab() -> DemoTab:

    with gr.Tab("Demo Patients"):

        with gr.Row(elem_classes=["main-pad"]):

            # ===========================================
            # Sidebar
            # ===========================================

            with gr.Column(
                scale=1,
                min_width=320,
                elem_classes=[
                    "sidebar-col",
                    "workspace-card"
                ]
            ):

                gr.HTML(get_sidebar_heading())

                demo_dropdown = gr.Dropdown(
                    label="Patient",
                    choices=[],
                    value=None,
                )

                demo_story = gr.HTML()

                analyse_btn = gr.Button(
                    "Analyse Patient",
                    variant="primary",
                    elem_id="analyse-btn",
                )

                gr.HTML(get_sidebar_stats())

                gr.HTML(get_disclaimer())

            # ===========================================
            # Main Panel
            # ===========================================

            with gr.Column(
                scale=3,
                elem_classes=["main-content"]
            ):

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
                    lines=4,
                    interactive=False,
                    visible=False
                )

                immune_plot = gr.Plot(
                    visible=False
                )

    return DemoTab(

        dropdown=demo_dropdown,

        story=demo_story,

        analyse_btn=analyse_btn,

        risk=risk_card,

        km=km_plot,

        shap=shap_plot,

        immune=immune_plot,

        explanation=explanation

    )