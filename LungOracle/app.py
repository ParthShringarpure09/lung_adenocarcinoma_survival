"""
LungOracle
AI-powered Lung Adenocarcinoma Survival Prediction
"""

import json
import warnings

import gradio as gr
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# UI
# ============================================================
from ui.result_panel import build_result_panel
from ui.css import get_css
from ui.header import get_header
from ui.footer import get_footer
from ui.dashboard import get_dashboard
from ui.sidebar import (
    get_sidebar_heading,
    get_sidebar_stats,
    get_disclaimer,
)
from ui.prediction import get_prediction_placeholder

# ============================================================
# ML Utilities
# ============================================================

from utils.inference import (
    load_models,
    run_inference,
)

from utils.shap_utils import (
    compute_shap_values,
    get_top_shap_features,
    get_shap_explanation_text,
)

from utils.survival_plot import (
    build_km_data,
    plot_survival_curve,
    plot_immune_profile,
    plot_shap_waterfall,
)

from utils.gradio_helpers import prediction_outputs

# ============================================================
# Data Paths
# ============================================================

DEMO_CSV = "data/demo_patients.csv"
ALL_RISK_JSON = "data/all_patient_risk_scores.json"

BASE_CLINICAL = (
    "/Users/parthshringarpure/Desktop/AI/Projects/"
    "luad_survival/data/processed/clinical_survival.csv"
)

BASE_EXPR = (
    "/Users/parthshringarpure/Desktop/AI/Projects/"
    "luad_survival/data/processed/expression_matrix.csv"
)

# ============================================================
# Load Models
# ============================================================

print("Loading models...")

MODELS = load_models()

print("Models loaded.")

# ============================================================
# Demo Patients
# ============================================================

DEMO_DF = pd.read_csv(DEMO_CSV)

DEMO_DF["short_name"] = DEMO_DF.apply(
    lambda r: (
        f"Patient {chr(65 + int(r.name))} — "
        f"{r['stage']}, "
        f"{int(r['age'])}yo, "
        f"{r['risk_group']} risk"
    ),
    axis=1,
)

DEMO_OPTIONS_SHORT = list(DEMO_DF["short_name"])

# ============================================================
# Risk Scores
# ============================================================

with open(ALL_RISK_JSON) as f:
    ALL_RISK_SCORES = json.load(f)

# ============================================================
# Kaplan–Meier Curves
# ============================================================

print("Building KM curves...")

KM_HIGH_DF, KM_LOW_DF = build_km_data(
    BASE_CLINICAL,
    ALL_RISK_SCORES,
)

print("KM curves ready.")

# ============================================================
# Expression Matrix
# ============================================================

print("Loading expression matrix...")

EXPR_MATRIX = pd.read_csv(
    BASE_EXPR,
    index_col=0,
)

EXPR_MATRIX.index = EXPR_MATRIX.index.str.upper()

print("LungOracle ready.")


def run_prediction(expr_series, age, gender, stage, survival_days, patient_label):
    result = run_inference(
        expr_input=expr_series,
        age=float(age), gender=gender, stage=stage, models=MODELS
    )

    if result["overlap_error"]:
        err = f"""
        <div style="background:#fff5f5;border-left:4px solid #c53030;
                    border-radius:0 8px 8px 0;padding:20px 24px;
                    font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;">
          <div style="font-size:13px;font-weight:700;color:#c53030;margin-bottom:6px;">
            Insufficient gene data
          </div>
          <div style="font-size:13px;color:#4a5568;">{result['overlap_message']}</div>
        </div>"""
        return err, None, None, None, ""

    risk_score  = result["risk_score"]
    risk_group  = result["risk_group"]
    immune_frac = result["immune_fractions"]

    col_map = {
        "HIGH":   ("#c53030", "#fed7d7", "#fc8181"),
        "MEDIUM": ("#c05621", "#feebc8", "#f6ad55"),
        "LOW":    ("#276749", "#c6f6d5", "#68d391"),
    }
    text_col, bg_col, border_col = col_map[risk_group]

    warn_parts = []
    if result["overlap_message"].startswith("WARNING"):
        warn_parts.append(result["overlap_message"])
    if result["stage_warning"]:
        warn_parts.append(result["stage_warning"])
    warn_str = " | ".join(warn_parts)

    m2_val  = immune_frac.get("Macrophages M2", 0)
    cd8_val = immune_frac.get("T cells CD8", 0)
    immuno  = "Immunosuppressive microenvironment" if m2_val > 0.15 else ""

    risk_html = f"""
    <div style="background:{bg_col};border:1.5px solid {border_col};
                border-left:5px solid {text_col};border-radius:8px;
                padding:24px 28px;
                font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;">
      <div style="display:flex;align-items:flex-start;gap:28px;flex-wrap:wrap;">
        <div style="min-width:120px;">
          <div style="font-size:10px;font-weight:700;letter-spacing:1.2px;
                      text-transform:uppercase;color:{text_col};margin-bottom:4px;">
            Predicted risk
          </div>
          <div style="font-size:56px;font-weight:800;color:{text_col};
                      line-height:1;margin-bottom:6px;letter-spacing:-1px;">
            {risk_group}
          </div>
          <div style="font-size:22px;font-weight:700;color:{text_col};">
            {risk_score:.3f}
          </div>
        </div>
        <div style="flex:1;min-width:200px;padding-top:2px;">
          <div style="font-size:13px;color:#4a5568;line-height:2.2;margin-bottom:12px;">
            <span style="color:#718096;">Age</span>&nbsp;
            <span style="font-weight:600;color:#2d3748;">{int(age)} years</span>
            &nbsp;&nbsp;
            <span style="color:#718096;">Sex</span>&nbsp;
            <span style="font-weight:600;color:#2d3748;">{gender}</span>
            &nbsp;&nbsp;
            <span style="color:#718096;">Stage</span>&nbsp;
            <span style="font-weight:600;color:#2d3748;">{stage}</span>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <span style="background:#fff5f5;color:#c53030;font-size:12px;
                         font-weight:600;padding:5px 12px;border-radius:4px;
                         border:1px solid #fc8181;">
              M2 macrophages &nbsp;{m2_val:.1%}
            </span>
            <span style="background:#f0fff4;color:#276749;font-size:12px;
                         font-weight:600;padding:5px 12px;border-radius:4px;
                         border:1px solid #68d391;">
              CD8 T cells &nbsp;{cd8_val:.1%}
            </span>
            {"<span style='font-size:12px;color:#c53030;font-style:italic;'>" + immuno + "</span>" if immuno else ""}
          </div>
        </div>
      </div>
      <div style="margin-top:16px;padding-top:12px;border-top:1px solid {border_col};
                  font-size:11px;color:#718096;">
        Thresholds: &nbsp;
        <span style="color:#276749;font-weight:700;">LOW</span> &lt;0.4
        &nbsp;|&nbsp;
        <span style="color:#c05621;font-weight:700;">MEDIUM</span> 0.4–0.6
        &nbsp;|&nbsp;
        <span style="color:#c53030;font-weight:700;">HIGH</span> &gt;0.6
        {"&nbsp;&nbsp;<span style='color:#c05621;'>&#9888; " + warn_str + "</span>" if warn_str else ""}
      </div>
    </div>"""

    km_fig = plot_survival_curve(
        KM_HIGH_DF, KM_LOW_DF,
        patient_risk_score=risk_score,
        patient_survival_days=survival_days if survival_days else 0,
        patient_risk_group=risk_group,
        patient_label=patient_label
    )

    shap_vals, base_val = compute_shap_values(
        result["feature_vector"], MODELS["xgb"], MODELS["scaler"],
        n_background=100
    )
    top_features = get_top_shap_features(result["feature_vector"], shap_vals, n_top=10)
    explanation  = get_shap_explanation_text(top_features)
    shap_fig     = plot_shap_waterfall(top_features)
    immune_fig   = plot_immune_profile(immune_frac, patient_label=patient_label)

    return risk_html, km_fig, shap_fig, immune_fig, explanation


def run_demo(short_name):
    if not short_name:
        return get_prediction_placeholder(), None, None, None, ""
    row = DEMO_DF[DEMO_DF["short_name"] == short_name].iloc[0]
    patient_id    = row["patient_id"]
    age           = row["age"]
    gender        = row["gender"]
    stage         = row["stage"]
    survival_days = row["survival_days"]
    if patient_id not in EXPR_MATRIX.index:
        return (
            f"<div style='color:#c53030;padding:20px;font-family:Inter,Arial,sans-serif;'>"
            f"Expression data not found for {patient_id}</div>",
            None, None, None, ""
        )
    expr_series = EXPR_MATRIX.loc[patient_id]
    short_label = short_name.split(" — ")[0]
    return run_prediction(expr_series, age, gender, stage, survival_days, short_label)


def run_predict(file_obj, age, gender, stage):
    if file_obj is None:
        return get_prediction_placeholder(), None, None, None, ""
    try:
        try:
            df = pd.read_csv(file_obj, sep=None, engine="python",
                             index_col=0, header=None)
            expr_series = df.iloc[:, 0].astype(float)
        except Exception:
            df = pd.read_csv(file_obj, sep="\t", index_col=0, header=None)
            expr_series = df.iloc[:, 0].astype(float)
        expr_series.index = expr_series.index.astype(str).str.strip()
    except Exception as e:
        return (
            f"<div style='background:#fff5f5;border-left:4px solid #c53030;"
            f"border-radius:0 8px 8px 0;padding:20px 24px;"
            f"font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;'>"
            f"<div style='font-size:13px;font-weight:700;color:#c53030;"
            f"margin-bottom:6px;'>File parse error</div>"
            f"<div style='font-size:13px;color:#4a5568;'>{str(e)}</div></div>",
            None, None, None, ""
        )
    return run_prediction(
        expr_series=expr_series, age=age, gender=gender, stage=stage,
        survival_days=0, patient_label="Uploaded patient"
    )


def update_story(short_name):
    if not short_name:
        return ""
    row = DEMO_DF[DEMO_DF["short_name"] == short_name].iloc[0]
    risk_col = {
        "HIGH": "#c53030", "MEDIUM": "#c05621", "LOW": "#276749"
    }.get(str(row["risk_group"]), "#718096")
    event_text = "Deceased" if row["event"] == 1 else "Alive / censored"
    surv_years = round(row["survival_days"] / 365.25, 1)
    return f"""
    <div style="background:#ffffff;border:1.5px solid #e2e8f0;border-radius:8px;
                padding:14px 16px;margin-top:8px;
                font-family:-apple-system,BlinkMacSystemFont,Inter,Arial,sans-serif;">
      <div style="font-size:15px;font-weight:700;color:#1a365d;margin-bottom:4px;">
        {short_name.split(' — ')[0]}
      </div>
      <div style="font-size:12px;color:#718096;margin-bottom:8px;">
        {row['stage']} &nbsp;·&nbsp; {int(row['age'])} years &nbsp;·&nbsp; {row['gender']}
      </div>
      <div style="font-size:12px;color:#4a5568;line-height:1.5;margin-bottom:10px;">
        {row['label']}
      </div>
      <div style="display:flex;gap:16px;padding-top:8px;border-top:1px solid #e2e8f0;">
        <div style="font-size:11px;color:#718096;">
          Survival&nbsp;<span style="font-weight:700;color:#2d3748;">{surv_years}y</span>
        </div>
        <div style="font-size:11px;color:#718096;">
          Outcome&nbsp;<span style="font-weight:700;color:#2d3748;">{event_text}</span>
        </div>
        <div style="font-size:11px;color:#718096;">
          Risk&nbsp;<span style="font-weight:700;color:{risk_col};">{row['risk_group']}</span>
        </div>
      </div>
    </div>"""

def run_demo_full(short_name):
    return prediction_outputs(
        *run_demo(short_name)
    )


def run_predict_full(file_obj, age, gender, stage):
    return prediction_outputs(
        *run_predict(
            file_obj,
            age,
            gender,
            stage,
        )
    )


def build_app():
    with gr.Blocks(
        css=get_css(),
        title="LungOracle — LUAD Survival Prediction",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        )
    ) as app:

        gr.HTML(get_header())

        with gr.Tabs():

            # ── DEMO MODE ──────────────────────────────────────────────────
            with gr.Tab("Demo Patients"):
                with gr.Row(elem_classes=["main-pad"]):

                    with gr.Column(scale=1, min_width=320, elem_classes=["sidebar-col"]):
                        gr.HTML(get_sidebar_heading("Patient Selection"))
                        demo_dropdown = gr.Dropdown(
                            choices=DEMO_OPTIONS_SHORT,
                            label="Select patient",
                            value=None,
                            info="10 patients — select to load instantly"
                        )
                        demo_story = gr.HTML(value="")
                        gr.HTML("<div style='height:10px;'></div>")
                        demo_btn = gr.Button(
                            "Analyse Patient",
                            elem_id="analyse-btn"
                        )
                        gr.HTML(get_sidebar_stats())

                        gr.HTML(get_disclaimer())

                    with gr.Column(scale=3):
                            (
                                demo_risk_card,
                                demo_km_plot,
                                demo_shap_plot,
                                demo_immune_plot,
                                demo_explanation,
                            ) = build_result_panel()

                demo_dropdown.change(
                    fn=update_story,
                    inputs=[demo_dropdown],
                    outputs=[demo_story]
                )

                def run_predict_full(file_obj, age, gender, stage):

                    return prediction_outputs(
                        *run_predict(
                            file_obj,
                            age,
                            gender,
                            stage,
                        )
                    )

                demo_btn.click(
                    fn=run_demo_full,
                    inputs=[demo_dropdown],
                    outputs=[demo_risk_card, demo_km_plot, demo_shap_plot,
                             demo_immune_plot, demo_explanation]
                )

            # ── PREDICT MODE ───────────────────────────────────────────────
            with gr.Tab("Predict Mode"):
                with gr.Row(elem_classes=["main-pad"]):

                    with gr.Column(scale=1, min_width=320, elem_classes=["sidebar-col"]):
                        gr.HTML(get_sidebar_heading("Clinical Inputs"))
                        predict_file = gr.File(
                            label="Gene expression file (CSV or TXT)",
                            file_types=[".csv", ".txt"]
                        )
                        predict_age = gr.Number(
                            label="Age (years)",
                            value=65, minimum=18, maximum=100
                        )
                        predict_gender = gr.Radio(
                            choices=["Male", "Female"],
                            label="Sex", value="Male"
                        )
                        predict_stage = gr.Dropdown(
                            choices=["Stage I", "Stage II", "Stage III",
                                     "Stage IV", "Unknown"],
                            label="Cancer stage", value="Stage I",
                            info="Select Unknown if stage is not available"
                        )
                        gr.HTML("<div style='height:10px;'></div>")
                        predict_btn = gr.Button(
                            "Analyse Patient",
                            elem_id="predict-analyse-btn"
                        )
                        gr.HTML(get_sidebar_stats())

                        gr.HTML(get_disclaimer())

                    with gr.Column(scale=3, elem_classes=["main-content"]):
                            (
                                predict_risk_card,
                                predict_km_plot,
                                predict_shap_plot,
                                predict_immune_plot,
                                predict_explanation,
                            ) = build_result_panel()

                def run_predict_full(file_obj, age, gender, stage):
                    risk_html, km, shap, immune, expl = run_predict(
                        file_obj, age, gender, stage
                    )
                    has = km is not None
                    return (
                        risk_html,
                        gr.update(value=km,     visible=has),
                        gr.update(value=shap,   visible=has),
                        gr.update(value=immune, visible=has),
                        gr.update(value=expl,   visible=has),
                    )

                predict_btn.click(
                    fn=run_predict_full,
                    inputs=[predict_file, predict_age,
                            predict_gender, predict_stage],
                    outputs=[predict_risk_card, predict_km_plot,
                             predict_shap_plot, predict_immune_plot,
                             predict_explanation]
                )

    return app
# ============================================================
# Launch Application
# ============================================================

if __name__ == "__main__":

    app = build_app()

    app.launch(
        server_name="127.0.0.1",   # or "0.0.0.0"
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=["assets"],

    )