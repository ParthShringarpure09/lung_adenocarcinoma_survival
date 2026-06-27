"""
Business logic for Demo Patients.

This module performs inference and prepares everything
required by the UI.

No Gradio components should be imported here.
"""

from typing import Tuple

import matplotlib.figure as mpl_fig


# ------------------------------------------------------------
# Internal imports
# ------------------------------------------------------------

# These will come from your existing project
#
# Replace the imports below with your real functions.
#

# from utils.inference import predict_patient
# from utils.shap_utils import create_shap_plot
# from utils.survival_plot import create_km_plot
# from utils.html import build_risk_card


# ============================================================
# Patient Story
# ============================================================

def update_patient_story(patient_id: str):

    """
    Returns the patient story shown in the sidebar.
    """

    if patient_id is None:

        return ""

    #
    # Replace with your current implementation.
    #

    return f"""
    <div class="lo-card" style="padding:18px;">

        <b>{patient_id}</b>

        <br><br>

        Patient summary appears here.

    </div>
    """


# ============================================================
# Main Prediction
# ============================================================

def analyse_demo_patient(patient_id: str):

    """
    Complete demo prediction workflow.

    Returns exactly the outputs expected by
    demo_callbacks.py
    """

    #
    # ------------------------------------------------------
    # Everything below should be replaced with your
    # existing prediction code.
    # ------------------------------------------------------
    #

    #
    # 1
    # Load patient
    #

    # patient = ...

    #
    # 2
    # Run model
    #

    # result = predict_patient(patient)

    #
    # 3
    # Build HTML
    #

    risk_html = """
    <div class="lo-card" style="padding:32px;">
        Prediction Complete
    </div>
    """

    #
    # 4
    # Create figures
    #

    km_plot = mpl_fig.Figure()

    shap_plot = mpl_fig.Figure()

    immune_plot = mpl_fig.Figure()

    #
    # 5
    # Explanation
    #

    explanation = """
Prediction successfully generated.

Replace this text with the real model interpretation.
"""

    #
    # Return order must match callback outputs.
    #

    return (

        risk_html,

        km_plot,

        shap_plot,

        explanation,

        immune_plot,

    )