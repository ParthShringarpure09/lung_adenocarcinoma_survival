"""
Prediction workspace components.

Contains all HTML used before and after inference.
"""

from ui.components import (
    info_card,
    metric_card,
    bullet,
)


def get_empty_prediction():

    html = []

    html.append("""
<div class="lo-dashboard">

<div style="padding:42px;">
""")

    # -------------------------------------------------
    # Header
    # -------------------------------------------------

    html.append("""

<div class="lo-kicker">

Prediction Dashboard

</div>

<div class="lo-title" style="font-size:34px;">

Ready to Analyse

</div>

<div class="lo-description">

Select a patient from the left panel or upload an
RNA-seq expression profile to generate a personalised
survival prediction.

</div>

""")

    # -------------------------------------------------
    # What the model generates
    # -------------------------------------------------

    html.append("""

<br>

<div class="lo-grid-2">

""")

    html.append(

        info_card(

            title="Survival Prediction",

            icon="📈",

            body="""

✓ Personalised risk score<br>

✓ Risk stratification<br>

✓ Survival probability

"""

        )

    )

    html.append(

        info_card(

            title="Model Explainability",

            icon="🧠",

            body="""

✓ SHAP values<br>

✓ Top molecular drivers<br>

✓ Clinical interpretation

"""

        )

    )

    html.append("""

</div>

""")

    # -------------------------------------------------
    # Workflow
    # -------------------------------------------------

    html.append("""

<br><br>

<div class="lo-card" style="padding:30px;">

<h3 style="margin-top:0;">

Prediction Workflow

</h3>

""")

    html.append(

        bullet("RNA-seq expression profile")

    )

    html.append(

        bullet("Feature engineering")

    )

    html.append(

        bullet("Ensemble machine learning")

    )

    html.append(

        bullet("Risk prediction")

    )

    html.append(

        bullet("SHAP explainability")

    )

    html.append("""

</div>

""")

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    html.append("""

<br><br>

<div class="lo-grid-4">

""")

    html.append(metric_card("478", "Patients"))

    html.append(metric_card("124", "Features"))

    html.append(metric_card("22", "Immune Cells"))

    html.append(metric_card("0.702", "C-index"))

    html.append("""

</div>

""")

    # -------------------------------------------------
    # Footer
    # -------------------------------------------------

    html.append("""

<br>

<div style="
text-align:center;
font-size:13px;
color:#94a3b8;
padding-bottom:10px;
">

Waiting for prediction...

</div>

""")

    html.append("""

</div>

</div>

""")

    return "".join(html)


def get_prediction_placeholder():
    return """
    <div style="
        background:#ffffff;
        border:1px solid #e2e8f0;
        border-radius:12px;
        overflow:hidden;
        font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;
    ">

        <div style="padding:36px 32px;text-align:center;">

            <div style="
                font-size:18px;
                font-weight:700;
                color:#1a365d;
                margin-bottom:10px;
            ">
                Select a patient to begin
            </div>

            <div style="
                font-size:14px;
                color:#718096;
                line-height:1.6;
                max-width:500px;
                margin:0 auto 28px;
            ">
                LungOracle integrates gene expression, tumour dysregulation,
                immune microenvironment and clinical variables to generate
                a personalised survival prediction.
            </div>

            <div style="
                display:flex;
                gap:12px;
                justify-content:center;
                margin-bottom:28px;
            ">

                <div style="
                    background:#f8fafc;
                    border:1px solid #e2e8f0;
                    border-radius:10px;
                    padding:16px;
                    flex:1;
                ">
                    <div style="font-size:22px;font-weight:700;color:#1a365d;">0.702</div>
                    <div style="font-size:11px;color:#718096;">C-index</div>
                </div>

                <div style="
                    background:#f8fafc;
                    border:1px solid #e2e8f0;
                    border-radius:10px;
                    padding:16px;
                    flex:1;
                ">
                    <div style="font-size:22px;font-weight:700;color:#1a365d;">478</div>
                    <div style="font-size:11px;color:#718096;">Patients</div>
                </div>

                <div style="
                    background:#f8fafc;
                    border:1px solid #e2e8f0;
                    border-radius:10px;
                    padding:16px;
                    flex:1;
                ">
                    <div style="font-size:22px;font-weight:700;color:#1a365d;">124</div>
                    <div style="font-size:11px;color:#718096;">Features</div>
                </div>

                <div style="
                    background:#f8fafc;
                    border:1px solid #e2e8f0;
                    border-radius:10px;
                    padding:16px;
                    flex:1;
                ">
                    <div style="font-size:22px;font-weight:700;color:#1a365d;">4</div>
                    <div style="font-size:11px;color:#718096;">Validation Cohorts</div>
                </div>

            </div>

        </div>

        <div style="
            border-top:1px solid #f1f5f9;
            padding:28px;
            background:#fafbfc;
            text-align:center;
        ">

            <div style="
                font-size:11px;
                font-weight:700;
                text-transform:uppercase;
                letter-spacing:.8px;
                color:#94a3b8;
                margin-bottom:20px;
            ">
                Model Architecture
            </div>

            <img
                src="/gradio_api/file=assets/architecture.png"
                style="
                    width:100%;
                    max-width:900px;
                    border-radius:12px;
                    border:1px solid #e2e8f0;
                "
            >

            <div style="
                margin-top:18px;
                font-size:12px;
                color:#94a3b8;
            ">
                Select a patient to generate a personalised survival prediction.
            </div>

        </div>

    </div>
    """