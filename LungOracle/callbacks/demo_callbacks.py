"""
Demo tab callback registration.

This module contains only Gradio event wiring.
No prediction logic should live here.
"""

from ui.demo_tab import DemoTab
from services.demo_service import (
    analyse_demo_patient,
    update_patient_story,
)


def register_demo_callbacks(tab: DemoTab):
    """
    Register all callbacks for the Demo Patients tab.
    """

    # -------------------------------------------------------
    # Update patient story when dropdown changes
    # -------------------------------------------------------

    tab.dropdown.change(
        fn=update_patient_story,
        inputs=tab.dropdown,
        outputs=tab.story,
    )

    # -------------------------------------------------------
    # Run prediction
    # -------------------------------------------------------

    tab.analyse_btn.click(
        fn=analyse_demo_patient,
        inputs=tab.dropdown,
        outputs=[
            tab.risk,
            tab.km,
            tab.shap,
            tab.explanation,
            tab.immune,
        ],
    )