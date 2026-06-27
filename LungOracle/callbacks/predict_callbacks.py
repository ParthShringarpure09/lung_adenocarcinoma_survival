"""
Predict tab callback registration.

Contains only Gradio event wiring.
"""

from ui.predict_tab import PredictTab
from services.upload_service import run_predict


def register_predict_callbacks(tab: PredictTab):
    """
    Register callbacks for Predict Mode.
    """

    tab.predict_btn.click(
        fn=run_predict,
        inputs=tab.upload,
        outputs=[
            tab.risk,
            tab.km,
            tab.shap,
            tab.explanation,
            tab.immune,
        ],
    )